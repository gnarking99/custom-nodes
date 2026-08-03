# -*- coding: utf-8 -*-
"""Panel de control: la parte de Python es minima, el trabajo lo hace js/control_panel.js"""

from comfy_api.latest import io


class ControlPanel(io.ComfyNode):
    """Un unico panel que controla nodos REPARTIDOS por todo el grafo.

    El problema con `Fast Groups Bypasser` de rgthree es que opera sobre
    **grupos**: para poder apagar tres nodos a la vez hay que moverlos
    fisicamente al mismo grupo. Eso destroza la organizacion espacial del flujo y
    llena la pantalla de cables cruzados.

    Aqui la pertenencia se marca en el **titulo** del nodo, asi que cada nodo se
    queda donde tiene sentido que este:

    - `#sw:mascara`  -> aparece un interruptor llamado `mascara` en el panel, que
      enciende y apaga (bypass) todos los nodos que lleven esa etiqueta, esten
      donde esten.
    - `#ui`  -> los widgets de ese nodo se **reflejan** dentro del panel. El
      artista los toca desde aqui sin buscar el nodo.

    Un nodo puede llevar las dos. El titulo se edita con doble clic sobre el.

    Ejemplo real: para el modo mascara del flujo de Flux, pon `#sw:mascara` en los
    titulos de `SetLatentNoiseMask`, `DifferentialDiffusion` e
    `ImageCompositeMasked`. Quedan donde estan, y el panel los enciende juntos.

    Nota: este nodo **no se ejecuta**. No tiene entradas ni salidas: es
    exclusivamente interfaz. Ponlo arriba a la izquierda y no lo conectes a nada.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CP_ControlPanel",
            display_name="Control Panel",
            category="character-pipeline/artist",
            description=(
                "Panel unico de interruptores y widgets. Marca los nodos con "
                "#sw:nombre (interruptor) o #ui (reflejar widgets) en su titulo."
            ),
            inputs=[],
            outputs=[],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        return io.NodeOutput()
