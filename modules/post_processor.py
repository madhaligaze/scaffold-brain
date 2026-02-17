"""
StructuralPostProcessor — Пост-обработка конструкции.
======================================================
НАЗНАЧЕНИЕ v3.2:
  - Принимает "скелет" от A* (только стойки + ригели)
  - Добавляет диагонали (bracing) в шахматном порядке
  - Добавляет настилы (decks) на горизонтальные уровни
  - Возвращает полную спецификацию элементов для Physics

ПРАВИЛА LAYHER:
  - Диагонали: каждые 2 пролёта по горизонтали, через один ярус по вертикали
  - Настилы: на каждом горизонтальном ригеле
  - Без диагоналей конструкция = карточный домик

Используется в:
  - main.py → /session/model → после A*, перед Physics
"""
from __future__ import annotations

import math
from typing import Dict, List, Set, Tuple


class StructuralPostProcessor:
    """
    Преобразует минималистичный путь A* в конструкцию со всеми элементами.

    Пример:
        processor = StructuralPostProcessor()
        skeleton = astar.find_path(start, target)  # список сегментов
        full_structure = processor.process(skeleton)
        # full_structure содержит: standards, ledgers, diagonals, decks
    """

    # Стандартные параметры Layher
    LIFT_HEIGHT = 2.07   # Высота яруса (стойка 2.07 м — самая частая)
    LEDGER_STEP = 1.09   # Минимальный шаг ригелей

    def __init__(self, lift_height: float = None):
        """
        Args:
            lift_height: переопределить высоту яруса (по умолчанию 2.07 м)
        """
        self.LIFT_HEIGHT = lift_height or self.LIFT_HEIGHT

    def process(self, skeleton_elements: List[Dict]) -> List[Dict]:
        """
        ГЛАВНЫЙ КОНВЕЙЕР пост-обработки.

        Args:
            skeleton_elements: список сегментов от A*
                [{"start": (x,y,z), "end": (x,y,z), "type": "standard"|"ledger",
                  "length": float}, ...]

        Returns:
            Полная структура:
            [{"id": str, "type": str, "start": tuple, "end": tuple,
              "length": float, "weight": float}, ...]
        """
        if not skeleton_elements:
            return []

        structure = []

        # Этап 1: Преобразуем скелет в пронумерованные элементы
        for i, seg in enumerate(skeleton_elements):
            elem = {
                "id": f"sk_{i}",
                "type": seg.get("type", "ledger"),
                "start": seg["start"],
                "end": seg["end"],
                "length": seg.get("length", self._euclidean(seg["start"], seg["end"])),
                "weight": self._estimate_weight(seg),
            }
            structure.append(elem)

        node_map = self._build_node_map(structure)

        decks = self._add_decks(structure, node_map)
        structure.extend(decks)

        diagonals = self._add_diagonals(structure, node_map)
        structure.extend(diagonals)

        return structure

    def _add_decks(self, structure: List[Dict], node_map: Set[str]) -> List[Dict]:
        """
        Кладёт стальные настилы на горизонтальные ригели.

        Логика: Если есть ledger/transom → кладём deck той же длины.
        """
        _ = node_map
        decks = []
        deck_id = 0

        for el in structure:
            etype = el.get("type", "")
            if etype in ("ledger", "transom"):
                s = el["start"]
                e = el["end"]
                if abs(s[2] - e[2]) < 0.05:
                    decks.append(
                        {
                            "id": f"deck_{deck_id}",
                            "type": "deck",
                            "start": s,
                            "end": e,
                            "length": el["length"],
                            "weight": 18.0,
                        }
                    )
                    deck_id += 1

        return decks

    def _add_diagonals(self, structure: List[Dict], node_map: Set[str]) -> List[Dict]:
        """
        Ставит диагонали (раскосы) для жёсткости конструкции.
        """
        diagonals = []
        diag_id = 0

        verticals = [el for el in structure if el.get("type") in ("standard", "vertical")]

        processed_pairs: Set[Tuple] = set()

        for v in verticals:
            s = v["start"]
            x, y, z = s[0], s[1], s[2]

            neighbors = self._find_horizontal_neighbors(x, y, z, node_map)

            for nx, ny in neighbors:
                pair_id = tuple(sorted(((x, y, z), (nx, ny, z))))
                if pair_id in processed_pairs:
                    continue

                h = self.LIFT_HEIGHT
                has_top_1 = self._node_exists(x, y, z + h, node_map)
                has_top_2 = self._node_exists(nx, ny, z + h, node_map)

                if has_top_1 and has_top_2:
                    diag_start = (x, y, z)
                    diag_end = (nx, ny, z + h)
                    diag_len = self._euclidean(diag_start, diag_end)

                    diagonals.append(
                        {
                            "id": f"diag_{diag_id}",
                            "type": "diagonal",
                            "start": diag_start,
                            "end": diag_end,
                            "length": round(diag_len, 4),
                            "weight": 8.5,
                        }
                    )
                    diag_id += 1
                    processed_pairs.add(pair_id)

        return diagonals

    def _build_node_map(self, elements: List[Dict]) -> Set[str]:
        """Строит набор ключей узлов "x_y_z" для быстрого поиска."""
        nodes = set()
        for el in elements:
            nodes.add(self._k(el["start"]))
            nodes.add(self._k(el["end"]))
        return nodes

    def _node_exists(self, x: float, y: float, z: float, node_map: Set[str]) -> bool:
        return self._k((x, y, z)) in node_map

    @staticmethod
    def _k(coords: Tuple[float, ...]) -> str:
        """Хеш-ключ узла с округлением."""
        return f"{coords[0]:.2f}_{coords[1]:.2f}_{coords[2]:.2f}"

    def _find_horizontal_neighbors(self, x: float, y: float, z: float, node_map: Set[str]) -> List[Tuple[float, float]]:
        """Ищет соседние узлы на том же уровне z на стандартных расстояниях Layher."""
        layher_lens = [0.73, 1.09, 1.57, 2.07, 2.57, 3.07]
        neighbors = []

        for length in layher_lens:
            for dx, dy in [(length, 0), (-length, 0), (0, length), (0, -length)]:
                if self._node_exists(x + dx, y + dy, z, node_map):
                    neighbors.append((x + dx, y + dy))

        return neighbors

    @staticmethod
    def _euclidean(p1: Tuple[float, ...], p2: Tuple[float, ...]) -> float:
        """Евклидово расстояние между точками."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    @staticmethod
    def _estimate_weight(seg: Dict) -> float:
        """Оценка веса элемента на основе типа и длины."""
        etype = seg.get("type", "ledger")
        length = seg.get("length", 2.0)

        weights_per_meter = {
            "standard": 6.0,
            "vertical": 6.0,
            "ledger": 4.5,
            "transom": 4.5,
            "diagonal": 3.5,
        }
        kg_per_m = weights_per_meter.get(etype, 5.0)
        return round(kg_per_m * length, 2)
