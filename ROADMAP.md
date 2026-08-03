# Roadmap

Orden por relación valor/esfuerzo. Cada entrada dice qué problema real resuelve.

---

## v0.1 — hecho (sin probar en ComfyUI)

- `Character Profile` · `Mask Guard` · `Batch Match` · `Snap Resolution` · `Video Chunk` · `Sigmas Denoise` · `Canon Panel Builder` · `Prompt Template`

**Lo primero es probarlos.** Cargar ComfyUI, meterlos en el flujo de Flux, corregir lo que pete. Hasta que eso pase, el resto del roadmap es especulación.

---

## v0.2 — cerrar el bucle de automatización

### `VLM Diff` (envoltorio)
Un nodo que hable con `ComfyUI-QwenVL` o `ComfyUI-ThinkingLLM` y devuelva la línea correctiva ya limpia (sin preámbulos del tipo "Looking at the image, I can see…", que es lo que devuelven los VLM la mitad de las veces).

**Por qué no está en v0.1:** depende de qué pack acabes usando. Hazlo cuando lo hayas decidido con pruebas.

### `Region From Text`
VLM dice `"left hand"` → SAM3 segmenta → máscara. Cierra el círculo: detección automática de qué está mal **y dónde**.

Requiere `ComfyUI-RMBG` y verificar que la entrada de texto de `SAM3Segment` sea conectable. Es el paso que convierte el flujo en verdaderamente automático.

### `Canon Sampler`
Empaqueta el grupo de 12 nodos que genera el canon muestreando el LoRA (prompt neutro + batch + semilla fija) en uno solo.

---

## v0.3 — reproducibilidad

### `Pipeline Report`
Escribe un JSON al lado de cada salida con **todo** lo que se usó: denoise de cada pase, fuerzas de LoRA, semillas, salida cruda del VLM, prompt final, versión del perfil.

Iterando sobre tres flujos, la pregunta más cara es *"¿qué tenía puesto cuando salió bien?"*. Esto la contesta.

### `Profile Diff`
Compara dos ejecuciones y dice qué parámetro cambió. Útil cuando una salida es mejor y no sabes por qué.

---

## v0.4 — vídeo

### `Chunk Stitcher`
Une los trozos que produce `Video Chunk` mezclando el solape con una rampa, en vez de cortar en seco. Ahora mismo la unión se hace fuera de ComfyUI.

### `Temporal Guard`
Mide la deriva entre frames consecutivos y avisa si supera un umbral. El flicker en v2v casi siempre es exceso de denoise, y detectarlo automáticamente ahorra revisar el clip a ojo.

---

## Descartado, y por qué

**Un ControlNet para Flux.2 Klein.** El único union que existe (`FLUX.2-dev-Fun-Controlnet-Union`) es para Flux.2 **dev**. Entrenar uno para Klein no es un custom node, es un proyecto de investigación con presupuesto de GPU.

**Un nodo que "lea" un LoRA para saber qué le falta al personaje.** Un LoRA son deltas de rango bajo sobre matrices de pesos: no hay información semántica que extraer. La vía viable es muestrearlo, que es lo que hace `Canon Panel Builder`.

**Un scheduler propio para Flux.2.** `Sigmas Denoise` resuelve el problema real (control de denoise) sin tocar el cálculo del shift, que ya está bien. Reimplementarlo sería asumir mantenimiento a cambio de nada.
