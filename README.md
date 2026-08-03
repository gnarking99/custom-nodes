# ComfyUI Character Pipeline

Nodos para flujos de **recuperación de detalle de personajes con LoRA** en Flux.2 Klein, LTX-2.3 y Wan 2.2.

No es un pack de propósito general. Cada nodo aquí nace de un fallo concreto que costó una sesión de depuración en un flujo real. Si un nodo no tiene detrás una historia de "esto nos rompió", no entra.

---

## Principio de diseño

Los tres flujos comparten los mismos tres problemas:

1. **Fallos silenciosos.** Una máscara vacía, un conteo de frames desalineado, un `{diff}` sin sustituir. El flujo no falla: entrega algo incorrecto con toda naturalidad. Los nodos de este pack **prefieren romper ruidosamente** a entregar basura.
2. **La misma constante escrita en seis sitios.** El nombre del LoRA, su fuerza, la palabra de disparo, la semilla del canon. Cambiar de personaje era tocar una docena de widgets.
3. **Reglas por modelo memorizadas a mano.** Múltiplo de 32 en Flux, 64 en LTX, 16 en Wan. Frames 8n+1 en LTX, 4n+1 en Wan. Equivocarse cuesta una ejecución entera.

---

## Los nodos

### `Character Profile` — organización
Carga la ficha de un personaje desde `profiles/<nombre>.json`: palabra de disparo, LoRA, fuerza en el pase de edición y en el de refinado, semilla del canon, prompt del canon y candado de identidad.

**El problema:** cambiar de personaje significaba editar los mismos valores en tres flujos distintos y equivocarse en alguno.

**Después:** un desplegable. Y el perfil es un JSON versionable — queda historial de qué fuerza funcionaba para cada personaje.

> Los perfiles se leen al arrancar. Si añades uno nuevo, reinicia ComfyUI.

### `Mask Guard` — guarda
Comprueba que una máscara tiene contenido útil antes de dejarla pasar. Devuelve `coverage` y `is_suspicious`, y puede detener la ejecución con un mensaje que explica la causa.

**El problema:** `LoadImage` devuelve una máscara **64×64 de ceros** cuando el PNG no tiene canal alfa. Esa máscara pasa por `InvertMask`, sale toda blanca, y el `ImageCompositeMasked` pega la imagen original encima del resultado. El flujo entrega el original y no avisa. Perdimos una sesión entera buscando el fallo en el sampler.

Detecta ese caso concreto y lo dice: *"64×64 y vacía es lo que devuelve LoadImage sin canal alfa; el MaskEditor no llegó al nodo"*.

También detecta el caso contrario (cobertura > 99%), que casi siempre es una inversión accidental.

### `Batch Match` — guarda
Iguala el número de frames de dos lotes.

**El problema:** `IndexError: index 49 is out of bounds for dimension 0 with size 49` en `ColorMatch`. La referencia tenía 49 frames y el resultado 97, porque el número de frames se fijaba a mano en dos sitios distintos.

### `Snap Resolution` — geometría
Escala a N megapíxeles **y** alinea las dimensiones al múltiplo que exige el modelo, en un nodo.

**El problema:** veníamos arrastrando un round-trip `VAEEncode → VAEDecode → GetImageSize` cuyo único propósito era obtener medidas legales. Cuesta VRAM y tiempo por nada. Más la aritmética mental de acordarse del múltiplo según el flujo.

### `Video Chunk` — geometría
Trocea un lote de frames en ventanas válidas, con solape, y expone la **longitud real** del trozo.

**El problema:** el flujo pedía 97 frames, el clip tenía 49. El troceador devolvía 49 imágenes mientras el latente seguía generando 97. La causa de fondo: la longitud se fijaba a mano en dos sitios.

Aquí sale de **una sola fuente**. Conéctala al latente de vídeo y al de audio, y el desalineamiento deja de ser posible.

> Existe [`IAMCCS LTX-2 Frame Count Validator`](https://www.runcomfy.com/comfyui-nodes/IAMCCS-nodes/iamccs-ltx2-frame-count-validator), que valida el **número**. Este además hace el corte y garantiza que imágenes y longitud salgan del mismo cálculo, que es donde estaba el fallo.

### `Sigmas Denoise` — muestreo
Recorta una curva de sigmas ya calculada para conseguir denoise parcial.

**El problema:** `Flux2Scheduler` calcula el shift en función de la resolución (que es lo que quieres) pero **no tiene entrada de denoise**: siempre devuelve la curva completa. Nos obligó a elegir entre shift correcto y control de estructura, y nos costó dos rondas de "por qué me cambia la cara".

Opera sobre el tensor `SIGMAS`, así que funciona con cualquier scheduler.

| denoise | efecto |
|---|---|
| 0.25-0.32 | apenas mueve nada |
| 0.38-0.45 | detalle con deriva mínima |
| 0.60+ | empieza a cambiar pose y expresión |
| 1.0 | regenera desde ruido |

### `Canon Panel Builder` — canon
Monta `[CANON 1 | CANON 2 | … | TARGET]` en una imagen con **etiquetas quemadas**, y genera la instrucción del VLM sincronizada con el número de paneles.

**Por qué es un nodo y no una cadena de `ImageConcanate`:** casi ningún nodo VLM acepta varias imágenes sueltas, así que hay que pegarlas. Pero concatenar a ciegas deja al modelo adivinando cuál es cuál. Con las etiquetas puede referirse a ellas sin ambigüedad.

Y sobre todo: **la instrucción y el número de paneles tienen que ir sincronizados**. Si subes el batch de canon a 4 y el texto sigue diciendo "dos paneles", el filtro anti-alucinación deja de funcionar en silencio.

> El filtro anti-alucinación es el motivo de usar varias muestras: una generación es una *muestra*, no una especificación. Si una saca cuatro dedos por accidente, exigir coincidencia entre todas lo descarta.

### `Prompt Template` — texto
Plantilla con variables con nombre y limpieza de huecos.

**El problema:** `StringReplace` del core sustituye una variable. Para montar "trigger + diferencias + candado de identidad" acabas encadenando tres nodos y perdiendo de vista el prompt real. Y si una variable no llega, te queda `{diff}` literal dentro del prompt — fallo que se cuela sin avisar.

Además entiende la respuesta `no changes` del VLM: elimina la variable en vez de escribirla, y expone `has_changes` para poder saltarse el pase entero.

---

## Instalación

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/gnarking99/custom-nodes.git comfyui-character-pipeline
```

Requiere **ComfyUI ≥ 0.23** (API de nodos V3). Sin dependencias más allá de las que ya trae ComfyUI (`torch`, `numpy`, `Pillow`).

## Perfiles

```
profiles/
  clayantboss.json
```

```json
{
  "trigger": "clayantboss",
  "lora_name": "Clayant_V01_000001900.safetensors",
  "strength_edit": 0.85,
  "strength_refine": 0.25,
  "canon_seed": 1234567890,
  "canon_prompt": "character model sheet, full body, neutral A-pose, hands visible with fingers spread, plain grey background",
  "identity_lock": "Keep the character identity, pose, facial expression, camera angle and composition exactly as in the reference image. Change nothing else."
}
```

---

## Pruebas

```bash
python tests/test_logic.py
```

Prueba la lógica pura de `PromptTemplate`, `VideoChunk` y `SnapResolution` **sin necesitar ComfyUI** (stubbea `comfy_api`). 19 comprobaciones. Úsalo antes de cada push.

Para probar los 8 nodos dentro de ComfyUI, carga `example_workflows/lab_all_nodes.json`: no requiere ningún modelo.

---

## Antes de fiarte: tres cosas que verificar en el primer arranque

Este pack está escrito contra la API V3 documentada, pero **no se ha ejecutado dentro de ComfyUI todavía**. Tres puntos concretos a comprobar y corregir si hace falta:

1. **`io.Sigmas`** — no aparece en la lista de tipos documentada explícitamente. Si `SigmasDenoise` falla al cargar, sustitúyelo por el equivalente `io.Custom("SIGMAS")`.
2. **`force_input=True`** en `io.String.Input` (usado en `PromptTemplate`) — es la forma V1 de forzar que un widget sea entrada. Verifica el nombre del parámetro en V3.
3. **`multiline=True`** en `io.String.Input` — mismo caso.

Si alguno peta, el mensaje de ComfyUI dice exactamente qué parámetro no existe. Son correcciones de una línea.

## Estado

Alfa. Ver [ROADMAP.md](ROADMAP.md).
