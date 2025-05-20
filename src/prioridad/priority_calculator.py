import pandas as pd

class PriorityCalculator:
    def __init__(
        self,
        client_map: dict[str, float],
        prop_map:   dict[str, float],
        inc_map:    dict[str, float],
        mod_map:    dict[str, float],
        orig_map:   dict[str, float],
        dest_map:   dict[str, float],
        w1: float, w2: float, w3: float
    ):
        # pesos globales (idealmente w1+w2+w3 == 1)
        self.w1, self.w2, self.w3 = w1, w2, w3
        # mapas precargados
        self.client_map   = client_map
        self.prop_map     = prop_map
        self.inc_map      = inc_map
        self.mod_map      = mod_map
        self.orig_map     = orig_map
        self.dest_map     = dest_map

    def calculate(
        self,
        cliente:     str,
        incoterm:    str,
        modalidad:   str,
        origin:      str,
        destination: str
    ) -> float:
        # 1) lookup de client & proportion
        c_w = self.client_map.get(cliente, 0.0)
        p_w = self.prop_map.get(cliente,   0.0)

        # 2) lookup de sub-pesos
        i_w = self.inc_map.get(incoterm,    0.0)
        m_w = self.mod_map.get(modalidad,   0.0)
        o_w = self.orig_map.get(origin,     0.0)
        d_w = self.dest_map.get(destination,0.0)

        complexity = (i_w + m_w + o_w + d_w) / 4

        return self.w1 * c_w + self.w2 * complexity + self.w3 * p_w