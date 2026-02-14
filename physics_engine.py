# physics_engine.py
from PyNiteFEA import FEModel3D
from models import ScaffoldStructure, AnalysisResult

class ScaffoldBrain:
    def __init__(self):
        # Характеристики стали (Сталь Ст3 или аналог для лесов)
        self.E = 210e9  # Модуль упругости (Па)
        self.G = 81e9   # Модуль сдвига (Па)
        self.nu = 0.3   # Коэффициент Пуассона
        # Предел текучести (когда металл начинает гнуться безвозвратно)
        self.yield_strength = 235e6 # 235 МПа

    def analyze(self, structure: ScaffoldStructure) -> list[AnalysisResult]:
        # 1. Создаем 3D модель
        model = FEModel3D()

        # 2. Добавляем узлы (Nodes)
        for node in structure.nodes:
            model.add_node(node.id, node.x, node.y, node.z)
            # Если это самый нижний узел (z=0), считаем его опорой (землей)
            if node.z <= 0.05: # Погрешность 5 см
                model.def_support(node.id, True, True, True, True, True, True) # Жесткая заделка

        # 3. Добавляем балки (Elements)
        for beam in structure.beams:
            # Тут нужно задать реальные сечения труб (J, Iy, Iz, A)
            # Для примера берем стандартную трубу 48х3 мм
            model.add_member(beam.id, beam.start_node_id, beam.end_node_id, 
                             self.E, self.G, 100, 200, 100, 0.0005) 

        # 4. Нагрузки (Loads)
        # Добавляем собственный вес конструкций (автоматически)
        # Добавляем полезную нагрузку (людей) на ригели
        for beam in structure.beams:
            if beam.type == "ledger": # Ригель
                # Давим вниз (-Z) распределенной нагрузкой
                model.add_member_dist_load(beam.id, 'Fz', -1000, -1000) # -1000 Н/м (~100 кг/м)

        # 5. РАСЧЕТ (Самое главное)
        try:
            model.analyze(check_stability=True) # Проверка устойчивости!
        except Exception as e:
            # Если конструкция падает (матрица вырождена), PyNite выдаст ошибку
            return [{"element_id": "GLOBAL", "load_percent": 100, "color": "BLACK", "warning": "COLLAPSE DETECTED"}]

        # 6. Интерпретация результатов
        results = []
        for beam in structure.beams:
            # Получаем максимальное напряжение в балке
            max_stress = model.GetMember(beam.id).max_stress() 
            
            # Считаем % загрузки: Текущее напряжение / Предел прочности
            load_ratio = abs(max_stress / self.yield_strength)
            
            # Определяем цвет
            color = "green"
            if load_ratio > 0.7: color = "yellow" # 70%
            if load_ratio > 0.95: color = "red"   # 95% (Опасно!)

            results.append(AnalysisResult(
                element_id=beam.id,
                load_percent=round(load_ratio * 100, 1),
                color=color
            ))
            
        return results