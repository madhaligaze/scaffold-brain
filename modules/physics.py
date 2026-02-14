from enum import Enum
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


class SupportType(str, Enum):
    FLOOR = "FLOOR"
    BEAM_CLAMP = "BEAM_CLAMP"
    SUSPENDED = "SUSPENDED"


class StructuralBrain:
    MATERIAL_NAME = "steel_s235"
    SECTION_NAME = "tube_48x3"

    def __init__(self):
        pass

    def _ensure_engine(self):
        if FEModel3D is None:
            raise RuntimeError(
                "PyNite engine is not installed. Install dependency with 'pip install pynitefea' "
                "or run 'pip install -r requirements.txt'."
            ) from _PYNITE_IMPORT_ERROR

    def _setup_materials(self, model):
        """Регистрирует материал и сечение для новых версий Pynite."""
        if hasattr(model, "materials") and self.MATERIAL_NAME not in getattr(model, "materials", {}):
            model.add_material(self.MATERIAL_NAME, E=2.1e11, G=8.1e10, nu=0.3, rho=7850, fy=235e6)

        if hasattr(model, "sections") and self.SECTION_NAME not in getattr(model, "sections", {}):
            model.add_section(self.SECTION_NAME, A=4.5e-4, Iy=1.1e-7, Iz=1.1e-7, J=2.2e-7)

    def create_model(self, nodes, beams, fixed_node_ids=None):
        """Создает базовую модель PyNite."""
        self._ensure_engine()
        model = FEModel3D()
        self._setup_materials(model)
        fixed_nodes = set(fixed_node_ids or [])

        for n in nodes:
            model.add_node(n['id'], n['x'], n['y'], n['z'])
            support_type = n.get('support_type', SupportType.FLOOR.value if n['z'] <= 0.05 else SupportType.SUSPENDED.value)
            if support_type == SupportType.FLOOR.value or n['id'] in fixed_nodes:
                model.def_support(n['id'], True, True, True, True, True, True)
            elif support_type == SupportType.BEAM_CLAMP.value:
                # Зажим на балке: фиксируем поступательные степени свободы, допускаем малый поворот
                model.def_support(n['id'], True, True, True, False, False, False)
            elif support_type == SupportType.SUSPENDED.value and n['id'] in fixed_nodes:
                model.def_support(n['id'], True, True, True, True, True, True)

        for b in beams:
            # Совместимость с новым и старым API Pynite
            try:
                model.add_member(
                    b['id'],
                    b['start'],
                    b['end'],
                    self.MATERIAL_NAME,
                    self.SECTION_NAME,
                )
            except TypeError:
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

    def _get_member(self, model, member_id):
        if hasattr(model, 'members') and member_id in model.members:
            return model.members[member_id]
        if hasattr(model, 'GetMember'):
            return model.GetMember(member_id)
        raise KeyError(f"Member '{member_id}' not found")

    def _estimate_member_stress(self, member):
        """Оценивает эквивалентное напряжение для версий без max_stress()."""
        if hasattr(member, 'max_stress'):
            return member.max_stress()

        area = max(getattr(member.section, 'A', 4.5e-4), 1e-9)
        iy = max(getattr(member.section, 'Iy', 1.1e-7), 1e-12)
        iz = max(getattr(member.section, 'Iz', 1.1e-7), 1e-12)
        # Полурадиус трубы 48 мм
        c = 0.024

        axial = abs(member.max_axial()) / area
        my = abs(member.max_moment('My')) * c / iy
        mz = abs(member.max_moment('Mz')) * c / iz

        return axial + max(my, mz)

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
            member = self._get_member(model, b['id'])
            stress = self._estimate_member_stress(member)
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
