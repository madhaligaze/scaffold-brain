"""
Enhanced Physics Engine with Closed-Loop Optimization
======================================================
СТАТУС: Non-negotiable (Обсуждению не подлежит)

Модуль отвечает за:
1. Расчет нагрузок с PyNite (FEM анализ)
2. Автоматическую пересборку конструкции при перегрузке (Closed Loop)
3. Умное добавление диагоналей вместо удаления балок

ПРАВИЛО: ИИ ОБЯЗАН пересобрать конструкцию ДО отправки пользователю,
         если нагрузка превышает 90%.
"""
import copy
import math
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

try:
    from PyNite import FEModel3D
except ImportError:
    try:
        from Pynite import FEModel3D
    except ImportError:
        FEModel3D = None

from layher_standards import LayherStandards, ComponentType


@dataclass
class LoadAnalysisResult:
    """Результат анализа нагрузок"""
    status: str  # "OK", "WARNING", "CRITICAL", "COLLAPSE"
    max_load_ratio: float  # Максимальная нагрузка (0.0 - 1.0+)
    beam_loads: List[Dict]  # Список {id, load_ratio, color, stress}
    critical_beams: List[str]  # ID критически нагруженных балок
    recommended_reinforcements: List[Dict]  # Рекомендации по усилению
    
    def is_safe(self) -> bool:
        """Конструкция безопасна?"""
        return self.status in ["OK", "WARNING"] and self.max_load_ratio < 1.0
    
    def needs_optimization(self) -> bool:
        """Требуется автоматическая оптимизация?"""
        return self.max_load_ratio >= LayherStandards.CRITICAL_LOAD_THRESHOLD


class StructuralBrain:
    """
    Расширенный движок физики с Closed-Loop оптимизацией.
    
    Не просто считает нагрузки, а ИСПРАВЛЯЕТ конструкцию автоматически.
    """
    
    def __init__(self):
        self.max_optimization_iterations = 5  # Предотвращение бесконечного цикла
        
    def _ensure_engine(self):
        """Проверка наличия PyNite"""
        if FEModel3D is None:
            raise RuntimeError(
                "PyNite не установлен. Установите: pip install pynitefea"
            )
    
    def create_model(self, nodes: List[Dict], beams: List[Dict], 
                    fixed_node_ids: Optional[Set[str]] = None) -> 'FEModel3D':
        """
        Создание FEM модели PyNite.
        
        Args:
            nodes: Список узлов [{id, x, y, z}, ...]
            beams: Список балок [{id, start, end, type?}, ...]
            fixed_node_ids: Множество ID узлов, которые закреплены
            
        Returns:
            Модель PyNite FEModel3D
        """
        self._ensure_engine()
        model = FEModel3D()
        fixed_nodes = set(fixed_node_ids or [])
        
        # Добавляем узлы
        for n in nodes:
            model.add_node(n['id'], n['x'], n['y'], n['z'])
            
            # Закрепляем узлы на земле (z <= 0.05м) или явно указанные
            if n['z'] <= 0.05 or n['id'] in fixed_nodes:
                # Полное закрепление (6 степеней свободы)
                model.def_support(n['id'], True, True, True, True, True, True)
        
        # Добавляем материал/сечение и балки с реальными свойствами Layher.
        # Поддерживаем обе сигнатуры PyNite/Pynite (старую и новую).
        G = LayherStandards.STEEL_YOUNGS_MODULUS / (2 * (1 + 0.3))
        material_name = "layher_steel"
        section_name = "layher_tube"

        if hasattr(model, 'materials') and material_name not in model.materials:
            model.add_material(
                material_name,
                E=LayherStandards.STEEL_YOUNGS_MODULUS,
                G=G,
                nu=0.3,
                rho=LayherStandards.STEEL_DENSITY,
                fy=LayherStandards.STEEL_YIELD_STRENGTH,
            )

        if hasattr(model, 'sections') and section_name not in model.sections:
            model.add_section(
                section_name,
                A=LayherStandards.PIPE_CROSS_SECTION_AREA,
                Iy=LayherStandards.PIPE_MOMENT_OF_INERTIA,
                Iz=LayherStandards.PIPE_MOMENT_OF_INERTIA,
                J=LayherStandards.PIPE_TORSION_CONSTANT,
            )

        for b in beams:
            try:
                model.add_member(
                    b['id'],
                    b['start'],
                    b['end'],
                    material_name,
                    section_name,
                )
            except TypeError:
                # Fallback для старых версий с E/G/A/I/J параметрами.
                model.add_member(
                    b['id'],
                    b['start'],
                    b['end'],
                    E=LayherStandards.STEEL_YOUNGS_MODULUS,
                    G=G,
                    Iy=LayherStandards.PIPE_MOMENT_OF_INERTIA,
                    Iz=LayherStandards.PIPE_MOMENT_OF_INERTIA,
                    J=LayherStandards.PIPE_TORSION_CONSTANT,
                    A=LayherStandards.PIPE_CROSS_SECTION_AREA,
                )

        return model
    
    def calculate_load_map(self, nodes: List[Dict], beams: List[Dict],
                          fixed_node_ids: Optional[Set[str]] = None,
                          distributed_load: float = -1000.0) -> LoadAnalysisResult:
        """
        Основной расчет нагрузок с цветовой индикацией.
        
        Args:
            nodes: Список узлов
            beams: Список балок
            fixed_node_ids: Закрепленные узлы
            distributed_load: Распределенная нагрузка в Н/м (отрицательная вниз)
            
        Returns:
            LoadAnalysisResult с детальным анализом
        """
        if not beams:
            return LoadAnalysisResult(
                status="OK",
                max_load_ratio=0.0,
                beam_loads=[],
                critical_beams=[],
                recommended_reinforcements=[]
            )
        
        # Создаем модель
        model = self.create_model(nodes, beams, fixed_node_ids)
        
        # Применяем нагрузки (гравитация + рабочая нагрузка)
        for b in beams:
            # Вертикальная распределенная нагрузка
            model.add_member_dist_load(b['id'], 'Fz', distributed_load, distributed_load)
        
        # Анализ
        try:
            model.analyze(check_stability=True)
        except Exception as e:
            # Конструкция неустойчива
            return LoadAnalysisResult(
                status="COLLAPSE",
                max_load_ratio=float('inf'),
                beam_loads=[],
                critical_beams=[bid['id'] for bid in beams],
                recommended_reinforcements=[{
                    "type": "add_supports",
                    "message": "Конструкция неустойчива! Требуются дополнительные опоры."
                }]
            )
        
        # Собираем результаты по каждой балке
        beam_loads = []
        critical_beams = []
        max_ratio = 0.0
        
        for b in beams:
            member = model.members[b['id']]
            
            # Получаем максимальное напряжение (максимум по всей длине балки)
            max_stress = abs(member.max_axial())  # Осевое напряжение
            
            # Вычисляем коэффициент использования (Unity Ratio)
            load_ratio = max_stress / LayherStandards.STEEL_YIELD_STRENGTH
            
            # Определяем цвет
            if load_ratio < LayherStandards.WARNING_LOAD_THRESHOLD:
                color = "green"
                status_text = "OK"
            elif load_ratio < LayherStandards.CRITICAL_LOAD_THRESHOLD:
                color = "yellow"
                status_text = "WARNING"
            else:
                color = "red"
                status_text = "CRITICAL"
                critical_beams.append(b['id'])
            
            beam_loads.append({
                "id": b['id'],
                "load_ratio": load_ratio,
                "color": color,
                "max_stress": max_stress,
                "status": status_text
            })
            
            max_ratio = float(max(max_ratio, load_ratio))
        
        # Генерируем рекомендации по усилению
        reinforcements = self._generate_reinforcements(
            nodes, beams, beam_loads, critical_beams
        )
        
        # Определяем общий статус
        if max_ratio >= 1.0:
            status = "COLLAPSE"
        elif max_ratio >= LayherStandards.CRITICAL_LOAD_THRESHOLD:
            status = "CRITICAL"
        elif max_ratio >= LayherStandards.WARNING_LOAD_THRESHOLD:
            status = "WARNING"
        else:
            status = "OK"
        
        return LoadAnalysisResult(
            status=status,
            max_load_ratio=max_ratio,
            beam_loads=beam_loads,
            critical_beams=critical_beams,
            recommended_reinforcements=reinforcements
        )
    
    def _generate_reinforcements(self, nodes: List[Dict], beams: List[Dict],
                                beam_loads: List[Dict], 
                                critical_beams: List[str]) -> List[Dict]:
        """
        Генерация рекомендаций по усилению конструкции.
        
        Returns:
            Список рекомендаций [{type, position?, message}, ...]
        """
        reinforcements = []
        
        if not critical_beams:
            return reinforcements
        
        # Анализируем критические балки
        node_map = {n['id']: n for n in nodes}
        
        for beam_id in critical_beams:
            beam = next((b for b in beams if b['id'] == beam_id), None)
            if not beam:
                continue
            
            load_info = next((bl for bl in beam_loads if bl['id'] == beam_id), None)
            if not load_info:
                continue
            
            start_node = node_map.get(beam['start'])
            end_node = node_map.get(beam['end'])
            
            if not start_node or not end_node:
                continue
            
            # Вычисляем длину и позицию балки
            dx = end_node['x'] - start_node['x']
            dy = end_node['y'] - start_node['y']
            dz = end_node['z'] - start_node['z']
            length = math.sqrt(dx**2 + dy**2 + dz**2)
            
            mid_x = (start_node['x'] + end_node['x']) / 2
            mid_y = (start_node['y'] + end_node['y']) / 2
            mid_z = (start_node['z'] + end_node['z']) / 2
            
            # Рекомендация: добавить диагональ
            reinforcements.append({
                "type": "add_diagonal",
                "beam_id": beam_id,
                "position": {"x": mid_x, "y": mid_y, "z": mid_z},
                "load_ratio": load_info['load_ratio'],
                "message": f"Добавить диагональ в секции с балкой {beam_id} "
                          f"(нагрузка {load_info['load_ratio']*100:.0f}%)"
            })
            
            # Если балка очень длинная, рекомендуем разбиение пролета
            if length > 2.5:
                reinforcements.append({
                    "type": "split_bay",
                    "beam_id": beam_id,
                    "position": {"x": mid_x, "y": mid_y, "z": 0.0},
                    "message": f"Разбить пролет балки {beam_id} (добавить промежуточную стойку)"
                })
        
        return reinforcements
    
    def optimize_structure_closed_loop(self, nodes: List[Dict], beams: List[Dict],
                                      fixed_node_ids: Optional[Set[str]] = None,
                                      target_safety: float = 0.85) -> Dict:
        """
        CLOSED LOOP ОПТИМИЗАЦИЯ - автоматическая пересборка конструкции.
        
        Алгоритм:
        1. Рассчитываем нагрузки
        2. Если макс. нагрузка > 90% → добавляем диагонали
        3. Пересчитываем
        4. Повторяем до тех пор, пока нагрузка < target_safety (85%)
        
        Args:
            nodes: Исходные узлы
            beams: Исходные балки
            fixed_node_ids: Закрепленные узлы
            target_safety: Целевой коэффициент безопасности (< 1.0)
            
        Returns:
            {
                "nodes": [...],  # Обновленные узлы
                "beams": [...],  # Обновленные балки (с добавленными диагоналями)
                "iterations": int,  # Количество итераций оптимизации
                "final_analysis": LoadAnalysisResult,
                "added_diagonals": int
            }
        """
        optimized_nodes = copy.deepcopy(nodes)
        optimized_beams = copy.deepcopy(beams)
        
        iteration = 0
        added_diagonals_total = 0
        
        while iteration < self.max_optimization_iterations:
            # Анализируем текущую конструкцию
            analysis = self.calculate_load_map(
                optimized_nodes, optimized_beams, fixed_node_ids
            )
            
            # Если конструкция безопасна — выходим
            if not analysis.needs_optimization():
                break
            
            # Если рухнула — выходим с ошибкой
            if analysis.status == "COLLAPSE":
                break
            
            # Добавляем диагонали согласно рекомендациям
            added_this_iteration = self._add_reinforcements(
                optimized_nodes, optimized_beams, analysis.recommended_reinforcements
            )
            
            added_diagonals_total += added_this_iteration
            
            # Если ничего не добавили — выходим (застряли)
            if added_this_iteration == 0:
                break
            
            iteration += 1
        
        # Финальный анализ
        final_analysis = self.calculate_load_map(
            optimized_nodes, optimized_beams, fixed_node_ids
        )
        
        return {
            "nodes": optimized_nodes,
            "beams": optimized_beams,
            "iterations": iteration,
            "final_analysis": final_analysis,
            "added_diagonals": added_diagonals_total,
            "optimized": bool(final_analysis.max_load_ratio < target_safety)
        }
    
    def _add_reinforcements(self, nodes: List[Dict], beams: List[Dict],
                           reinforcements: List[Dict]) -> int:
        """
        Применение рекомендаций по усилению (добавление диагоналей).
        
        Returns:
            Количество добавленных элементов
        """
        added_count = 0
        node_map = {n['id']: n for n in nodes}
        
        for rec in reinforcements:
            beam_id = rec.get('beam_id')
            if not beam_id:
                continue

            # Находим критическую балку
            beam = next((b for b in beams if b['id'] == beam_id), None)
            if not beam:
                continue

            start_node = node_map.get(beam['start'])
            end_node = node_map.get(beam['end'])

            if not start_node or not end_node:
                continue

            if rec['type'] == 'add_diagonal':
                # Добавляем диагональ от start к противоположному верхнему узлу
                diagonal_id = f"diag_{beam_id}_{added_count}"

                # Ищем верхний узел над start_node
                upper_node = self._find_upper_node(nodes, start_node)
                if upper_node and upper_node['id'] != end_node['id']:
                    beams.append({
                        "id": diagonal_id,
                        "start": start_node['id'],
                        "end": upper_node['id'],
                        "type": "diagonal"
                    })
                    added_count += 1

            elif rec['type'] == 'split_bay':
                # Разбиение пролета: добавляем промежуточную стойку и 2 коротких балки
                mid_x = (start_node['x'] + end_node['x']) / 2
                mid_y = (start_node['y'] + end_node['y']) / 2
                mid_z = start_node['z']

                support_node_id = f"split_{beam_id}_{added_count}"
                if support_node_id in node_map:
                    continue

                support_node = {"id": support_node_id, "x": mid_x, "y": mid_y, "z": mid_z}
                nodes.append(support_node)
                node_map[support_node_id] = support_node

                # Заменяем одну длинную балку двумя короткими
                beams.remove(beam)
                beams.append({
                    "id": f"{beam_id}_a",
                    "start": start_node['id'],
                    "end": support_node_id,
                    "type": beam.get('type', 'ledger')
                })
                beams.append({
                    "id": f"{beam_id}_b",
                    "start": support_node_id,
                    "end": end_node['id'],
                    "type": beam.get('type', 'ledger')
                })
                added_count += 1

        return added_count
    
    def _find_upper_node(self, nodes: List[Dict], ref_node: Dict) -> Optional[Dict]:
        """
        Находит узел выше заданного (для добавления диагонали).
        
        Returns:
            Узел выше или None
        """
        # Ищем узел с близкими x,y координатами, но z выше на ~0.5-3.0м
        for node in nodes:
            if node['id'] == ref_node['id']:
                continue
            
            dx = abs(node['x'] - ref_node['x'])
            dy = abs(node['y'] - ref_node['y'])
            dz = node['z'] - ref_node['z']
            
            # Узел должен быть примерно над ref_node
            if dx < 0.5 and dy < 0.5 and 0.5 < dz < 3.0:
                return node
        
        return None
    
    def simulate_removal(self, nodes: List[Dict], beams: List[Dict],
                        remove_id: str, fixed_node_ids: Optional[Set[str]] = None) -> Dict:
        """
        Моделирование удаления элемента (для демонтажа).
        
        Args:
            nodes: Узлы
            beams: Балки
            remove_id: ID балки для удаления
            fixed_node_ids: Закрепленные узлы
            
        Returns:
            {
                "safe": bool,
                "message": str,
                "analysis": LoadAnalysisResult?
            }
        """
        # Создаем копию без удаляемого элемента
        new_beams = [b for b in beams if b['id'] != remove_id]
        
        # Анализируем
        result = self.calculate_load_map(nodes, new_beams, fixed_node_ids)
        
        if result.status == "COLLAPSE":
            return {
                "safe": False,
                "message": "КРИТИЧЕСКАЯ ОШИБКА! Конструкция потеряет устойчивость!",
                "analysis": result
            }
        
        if result.max_load_ratio > 1.0:
            return {
                "safe": False,
                "message": f"ОПАСНО! Перегрузка до {result.max_load_ratio*100:.0f}%",
                "analysis": result
            }
        
        if result.max_load_ratio > LayherStandards.CRITICAL_LOAD_THRESHOLD:
            return {
                "safe": False,
                "message": f"ПРЕДУПРЕЖДЕНИЕ: Нагрузка возрастет до {result.max_load_ratio*100:.0f}%",
                "analysis": result
            }
        
        return {
            "safe": True,
            "message": f"Безопасно. Макс. нагрузка {result.max_load_ratio*100:.0f}%",
            "analysis": result
        }


# ═══════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════

def quick_safety_check(nodes: List[Dict], beams: List[Dict]) -> bool:
    """
    Быстрая проверка безопасности без полного FEM анализа.
    
    Проверяет:
    - Достаточно ли опор
    - Нет ли слишком длинных балок без поддержки
    - Базовая геометрическая устойчивость
    
    Returns:
        True если конструкция выглядит безопасной
    """
    # Проверка 1: Есть ли опоры на земле?
    ground_nodes = [n for n in nodes if n['z'] <= 0.05]
    if len(ground_nodes) < 3:
        return False
    
    # Проверка 2: Есть ли балки?
    if len(beams) < 3:
        return False
    
    # Проверка 3: Нет ли слишком длинных балок?
    node_map = {n['id']: n for n in nodes}
    for beam in beams:
        start = node_map.get(beam['start'])
        end = node_map.get(beam['end'])
        if not start or not end:
            continue
        
        dx = end['x'] - start['x']
        dy = end['y'] - start['y']
        dz = end['z'] - start['z']
        length = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Балки длиннее 4м без поддержки опасны
        if length > 4.0:
            return False
    
    return True


if __name__ == "__main__":
    print("🧪 ТЕСТИРОВАНИЕ PHYSICS ENGINE")
    print("=" * 70)
    
    # Простой тест: балка на двух опорах
    test_nodes = [
        {"id": "n1", "x": 0.0, "y": 0.0, "z": 0.0},
        {"id": "n2", "x": 2.07, "y": 0.0, "z": 0.0},
        {"id": "n3", "x": 0.0, "y": 0.0, "z": 2.0},
        {"id": "n4", "x": 2.07, "y": 0.0, "z": 2.0},
    ]
    
    test_beams = [
        {"id": "b1", "start": "n1", "end": "n3", "type": "standard"},
        {"id": "b2", "start": "n2", "end": "n4", "type": "standard"},
        {"id": "b3", "start": "n3", "end": "n4", "type": "ledger"},
    ]
    
    brain = StructuralBrain()
    
    print("\n1. Быстрая проверка безопасности:")
    is_safe = quick_safety_check(test_nodes, test_beams)
    print(f"   {'✓ БЕЗОПАСНО' if is_safe else '✗ ОПАСНО'}")
    
    if FEModel3D is not None:
        print("\n2. Полный FEM анализ:")
        result = brain.calculate_load_map(test_nodes, test_beams)
        print(f"   Статус: {result.status}")
        print(f"   Макс. нагрузка: {result.max_load_ratio*100:.1f}%")
        
        if result.needs_optimization():
            print("\n3. Запуск Closed Loop оптимизации:")
            optimized = brain.optimize_structure_closed_loop(test_nodes, test_beams)
            print(f"   Итераций: {optimized['iterations']}")
            print(f"   Добавлено диагоналей: {optimized['added_diagonals']}")
            print(f"   Финальная нагрузка: {optimized['final_analysis'].max_load_ratio*100:.1f}%")
    else:
        print("\n⚠️ PyNite не установлен. Полный тест пропущен.")
    
    print("\n" + "=" * 70)
    print("✓ Тесты завершены!")