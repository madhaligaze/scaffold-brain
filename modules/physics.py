import copy

try:
    # Актуальная точка входа в pynitefea>=2.x
    from Pynite import FEModel3D
except ModuleNotFoundError:
    try:
        # Поддержка старых окружений, где модуль назывался PyNite
        from PyNite import FEModel3D
    except ModuleNotFoundError as exc:
        FEModel3D = None
        _PYNITE_IMPORT_ERROR = exc
    else:
        _PYNITE_IMPORT_ERROR = None
else:
    _PYNITE_IMPORT_ERROR = None


class StructuralBrain:
    def __init__(self):
        pass

    def _ensure_engine(self):
        if FEModel3D is None:
            raise RuntimeError(
                "PyNite engine is not installed. Install dependency with 'pip install pynitefea' "
                "or run 'pip install -r requirements.txt'."
            ) from _PYNITE_IMPORT_ERROR

    def create_model(self, nodes, beams, fixed_node_ids=None):
        """Создает базовую модель PyNite."""
        self._ensure_engine()
        model = FEModel3D()
        fixed_nodes = set(fixed_node_ids or [])

        for n in nodes:
            model.add_node(n['id'], n['x'], n['y'], n['z'])
            if n['z'] <= 0.05 or n['id'] in fixed_nodes:
                # Земля или навесная опора
                model.def_support(n['id'], True, True, True, True, True, True)

        for b in beams:
            # Стандартная труба 48х3мм: J, Iy, Iz, A...
            model.add_member(
                b['id'],
                b['start'],
                b['end'],
                E=2.1e11,
                G=8.1e10,
                Iy=1.1e-7,
                Iz=1.1e-7,
                J=2.2e-7,
                A=4.5e-4,
            )
        return model

    def calculate_load_map(self, nodes, beams, fixed_node_ids=None):
        """Основной расчет. Возвращает раскраску (Зеленый/Красный)."""
        if not beams:
            return {"status": "OK", "data": []}

        model = self.create_model(nodes, beams, fixed_node_ids=fixed_node_ids)

        # Добавляем гравитацию и нагрузку
        for b in beams:
            model.add_member_dist_load(b['id'], 'Fz', -200, -200)  # Нагрузка

        try:
            model.analyze(check_stability=True)
        except Exception:
            return {"status": "COLLAPSE", "data": []}

        results = []
        for b in beams:
            # Получаем напряжения
            member = model.GetMember(b['id'])
            stress = member.max_stress()
            ratio = abs(stress / 235e6)  # Предел текучести стали

            color = "green"
            if ratio > 0.6:
                color = "yellow"
            if ratio > 0.9:
                color = "red"

            results.append({
                "id": b['id'],
                "load_ratio": ratio,
                "color": color,
                "max_stress": stress,
            })
        return {"status": "OK", "data": results}

    def simulate_removal(self, nodes, beams, remove_id, fixed_node_ids=None):
        """Моделирует удаление элемента и оценивает последствия."""
        # Создаем копию списка балок БЕЗ удаляемого элемента
        new_beams = [b for b in beams if b['id'] != remove_id]

        # Запускаем расчет
        result = self.calculate_load_map(nodes, new_beams, fixed_node_ids=fixed_node_ids)

        if result["status"] == "COLLAPSE":
            return {
                "safe": False,
                "message": "КРИТИЧЕСКАЯ ОШИБКА! Конструкция потеряет устойчивость!",
            }

        if not result["data"]:
            return {"safe": True, "message": "Безопасно. После удаления не осталось нагруженных балок."}

        # Проверяем перегрузки
        max_load = max([r['load_ratio'] for r in result['data']])
        if max_load > 1.0:
            return {
                "safe": False,
                "message": f"ОПАСНО! Балка перегрузится до {int(max_load * 100)}%",
            }

        return {"safe": True, "message": "Безопасно. Макс нагрузка перераспределится."}
