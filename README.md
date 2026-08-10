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

### `LoRA Trigger Reader` — organización
Saca la trigger word **del propio archivo del LoRA**. Sin preguntarle a nadie.

**El problema:** cada vez que llega un LoRA nuevo hay que escribirle a quien lo entrenó para preguntarle la palabra de disparo.

**Y esa información ya está dentro del archivo.** Un `.safetensors` empieza por 8 bytes con el tamaño de la cabecera, seguidos de un JSON con `__metadata__`. El nodo lee **solo esa cabecera** — no carga los pesos, así que tarda milisegundos aunque el LoRA pese 2 GB.

Busca por orden, y siempre dice **de dónde lo sacó**:

1. Claves explícitas: `modelspec.trigger_phrase`, `ss_activation_tags`, `ss_trigger_words`
2. `ss_tag_frequency` — el histograma de etiquetas del dataset. Un trigger aparece en prácticamente **todas** las imágenes; las demás etiquetas describen imágenes sueltas. La que roza el 100% es el trigger. Descarta stopwords (`1girl`, `masterpiece`…)
3. El nombre del archivo, como último recurso — y avisa de que es una suposición

**Bonus que evita un fallo caro:** detecta la arquitectura (`flux` / `ltx` / `wan` / `sdxl`…) y puede **detener la ejecución si no coincide** con la que esperas. Cargar un LoRA de Flux sobre LTX no da error: simplemente no aplica ninguna clave y te vuelves loco buscando por qué "no se nota".

> Existen nodos que hacen algo parecido ([Lora-Auto-Trigger-Words](https://github.com/idrirap/ComfyUI-Lora-Auto-Trigger-Words), [Lora-Manager](https://github.com/willmiao/ComfyUI-Lora-Manager)), pero tiran sobre todo de la **API de Civitai**. Para LoRAs entrenados en casa eso no sirve: este solo mira el archivo.

### `Quality Preset` — artista
Un desplegable en vez de seis números sueltos: **Borrador / Normal / Máximo / Solo limpiar**, por flujo (Flux, LTX, Wan).

Tus artistas no tienen por qué saber que `steps 14 × denoise 0.42` da 6 pasos reales, ni que Wan quiere 0.15-0.25 mientras Flux aguanta 0.42. La salida `resumen` está escrita para leerse en un `PreviewAny` y avisa si el ajuste los mete en zona peligrosa.

### `Pipeline Status` — artista
Un semáforo en castellano en vez de trazas de Python. Recoge lo que reportan las guardas y lo convierte en un párrafo legible: *"Máscara: cubre el 34.2% del encuadre"*, *"FALTAN TROZOS: sube chunk_index en 1"*, *"Solo 2 pasos de muestreo: es muy poco"*.

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

### `LTX Gen Plan` — geometría
El selector de **generación** en LTX-2.3: eliges modo, entrega y duración, y sale todo lo demás.

**El problema:** en un flujo de generación de LTX hay tres números que se escriben a mano en seis sitios y que, si no cuadran, **no dan error** — dan un vídeo deformado o un fallo raro a mitad de cola.

1. **La resolución base no es la entrega.** La etapa 1 genera a la mitad y el upscaler latente dobla. Poner la entrega en `EmptyLTXVLatentVideo` cuadruplica el coste del paso lento y saca al upscaler de su rango.
2. **`frames` tiene que ser 8n+1** — el VAE comprime 8× en el eje temporal — pero sale de duración × fps, que son los dos números que el artista sí quiere tocar. 5 s × 24 fps = 120, y lo válido de al lado es 121.
3. **El modo decide qué ramas del grafo están vivas.** Las salidas booleanas se cablean a los `bypass` y a los switches de grupo: cambiar de t2v a ref2v es mover **un** desplegable.

Dos detalles que evitan fallos concretos:

- **`bypass_imagen` está invertida a propósito.** `LTXVImgToVideoInplace` tiene una entrada llamada `bypass`, no `enable`. Cablear la booleana correcta evita el clásico *"he puesto i2v y me sigue ignorando la imagen"*.
- **Hay dos salidas de fps, `fps` (INT) y `fps_f` (FLOAT)**, porque LTX las pide de los dos tipos en el mismo flujo: `LTXVEmptyLatentAudio` quiere INT y `LTXVConditioning` / `CreateVideo` quieren FLOAT. Dos primitivas separadas es justo el duplicado que desincroniza el audio cuando cambias una y olvidas la otra.

> Es el hermano de `LTX Resolution`, que hace lo mismo para **v2v** (parte de un vídeo existente). Este parte de la nada.

### `LTX Sigmas` — muestreo
Las curvas de sigmas de LTX-2.3 en un desplegable, cada una **con su procedencia escrita**.

**El problema:** LTX-2.3 no usa `steps` + `denoise`, usa una lista explícita. Las listas viajan por los foros como cadenas sueltas, sin decir a qué etapa pertenecen ni de dónde salieron. En un flujo de dos etapas hay dos `ManualSigmas` con listas parecidas y acabas cambiando la equivocada.

Trae la curva oficial de las plantillas de Comfy-Org y Lightricks, la de **movimiento + LoRA** que la comunidad usa cuando el personaje sale bien pero se queda quieto, y las colas de la segunda etapa.

`inicio` reescala la curva entera para que arranque donde digas **conservando su forma** y el 0.0 final. Es la palanca de fidelidad de la etapa 2: es preferible a cambiar de sampler, porque un solver de segundo orden en la segunda pasada suele salir duro y lo correcto es bajar la sigma de arranque.

Y rompe ruidosamente si la curva no es estrictamente decreciente: un valor que sube da salidas ruidosas **sin ningún mensaje de error**.

### `Wan Gen Plan` — geometría
El equivalente de `LTX Gen Plan` para Wan 2.2, con una diferencia que lo cambia todo: **el troceado en tandas de 81**.

**El problema:** Wan 2.2 se entrenó con clips de 81 frames a 16 fps. Pasarse **no da error** — el vídeo entra en bucle o pierde coherencia. Los clips largos se hacen generando tandas y encadenándolas, y ahí hay cuatro cuentas que se hacen mal:

1. **Pedir más de 81 de una vez.** El nodo tapa la longitud y te dice cuántas tandas hacen falta.
2. **`length` tiene que ser 4n+1.** 80 no vale, 81 sí. Y 5 s × 16 fps = 80, así que el redondeo va al más cercano, no hacia abajo: lo que quieres ahí es 81, el número canónico de Wan.
3. **Resolución múltiplo de 16.**
4. **El solape se cuenta dos veces.** Con tandas de 81 y solape 16, cada tanda aporta **65 frames nuevos**. Cuatro tandas no son 20 s, son 16.3 s.

### `Wan High/Low Split` — muestreo
Parte la curva de sigmas en el punto donde Wan 2.2 cambia de experto.

**El problema:** Wan 2.2 es MoE y el traspaso de high-noise a low-noise no va por la mitad de los pasos, va a una **sigma** concreta: **0.9 en I2V, 0.875 en T2V**. Pero el paso en el que cae esa sigma depende del `shift`, del scheduler y del número de pasos. Cambias el shift de 5 a 8 y el paso correcto se mueve; `SplitSigmas` del core sigue cortando donde lo dejaste. No da error: te quedas muestreando con el experto equivocado en parte de la curva, y el síntoma —movimiento pobre, identidad que se va— parece un problema de LoRA.

Este nodo busca el corte por ti y expone dónde ha caído y con cuántos pasos a cada lado, para verlo antes de lanzar. El reparto **no sale 50/50** y eso es lo correcto: con shift 5 y 20 pasos, la sigma 0.9 cae hacia el paso 7.

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

## Perfiles: ¿hace falta un JSON por LoRA?

**No.** Y conviene entender el reparto:

| Dato | De dónde sale | ¿JSON? |
|---|---|---|
| **Trigger word** | `LoRA Trigger Reader` — lo lee del archivo | **No** |
| Arquitectura, dim, nº de imágenes | `LoRA Trigger Reader` | **No** |
| steps, denoise, megapíxeles | `Quality Preset` | **No** |
| Fuerzas que **tú** ajustaste a ojo | no se puede deducir | Sí |
| Prompt del canon | no se puede deducir | Sí |
| Candado de identidad | igual para todos | No (default del nodo) |

O sea: **`LoRA Trigger Reader` + `Quality Preset` cubren lo que antes obligaba a escribir un JSON.** El perfil solo tiene sentido cuando llevas tiempo con un personaje y quieres congelar las fuerzas concretas que te funcionaron.

Empieza sin perfiles. Créalos solo para los personajes que ya tengas afinados.

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
python tests/test_meta.py
python tests/test_res.py
python tests/test_gen.py
python tests/test_wan.py
```

Prueban la lógica pura **sin necesitar ComfyUI** (stubbean `comfy_api`). `test_logic` cubre `PromptTemplate`, `VideoChunk` y `SnapResolution`; `test_res` cubre `LTXResolution`; `test_gen` cubre `LTXGenPlan` y `LTXSigmas` (27 comprobaciones); `test_wan` cubre `WanGenPlan` y `WanSplit` (32: troceado en tandas, avance del solape, corte por sigma y cómo lo mueve el shift). Úsalos antes de cada push.

## Generador de flujos

```bash
python tools/build_ltx_gen.py
python tools/build_wan_gen.py
```

Generan `LTX23 GEN t2v-i2v-ref2v.json` y `WAN22 GEN t2v-i2v-ref2v.json`. Un flujo de ComfyUI guarda cada enlace **tres veces** (en `links`, en `outputs[].links` del origen y en `inputs[].link` del destino); editarlo a mano deja incoherencias que aparecen a mitad de una cola de 20 minutos. Las firmas de nodo salen de `tools/ltx_node_shapes.json` y `tools/wan_node_shapes.json`, extraídos de las plantillas oficiales de Comfy-Org y Lightricks, y el script valida tipos, slots, entradas obligatorias sueltas y solapes antes de escribir.

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
