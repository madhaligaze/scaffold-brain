# modules/physics.py
from PyNite import FEModel3D
import copy

class StructuralBrain:
    def __init__(self):
        pass

    def create_model(self, nodes, beams):
        """Создает базовую модель PyNite"""
        model = FEModel3D()
        for n in nodes:
            model.add_node(n['id'], n['x'], n['y'], n['z'])
            if n['z'] <= 0.05: # Земля
                model.def_support(n['id'], True, True, True, True, True, True)
        
        for b in beams:
            # Стандартная труба 48х3мм: J, Iy, Iz, A...
            model.add_member(b['id'], b['start'], b['end'], 
                             E=2.1e11, G=8.1e10, Iy=1.1e-7, Iz=1.1e-7, J=2.2e-7, A=4.5e-4)
        return model

    def calculate_load_map(self, nodes, beams):
        """
        Основной расчет. Возвращает раскраску (Зеленый/Красный)
        """
        model = self.create_model(nodes, beams)
        
        # Добавляем гравитацию и нагрузку
        for b in beams:
             model.add_member_dist_load(b['id'], 'Fz', -200, -200) # Нагрузка

        try:
            model.analyze(check_stability=True)
        except:
            return {"status": "COLLAPSE", "data": []}

        results = []
        for b in beams:
            # Получаем напряжения
            member = model.GetMember(b['id'])
            stress = member.max_stress()
            ratio = abs(stress / 235e6) # Предел текучести стали
            
            color = "green"
            if ratio > 0.6: color = "yellow"
            if ratio > 0.9: color = "red"
            
            results.append({
                "id": b['id'],
                "load_ratio": ratio,
                "color": color,
                "max_stress": stress
            })
        return {"status": "OK", "data": results}

    def simulate_removal(self, nodes, beams, remove_id):
        """
        Моделирует удаление элемента.
        Возвращает: "Упадет" или "Устоит".
        """
        # Создаем копию списка балок БЕЗ удаляемого элемента
        new_beams = [b for b in beams if b['id'] != remove_id]
        
        # Запускаем расчет
        result = self.calculate_load_map(nodes, new_beams)
        
        if result["status"] == "COLLAPSE":
            return {
                "safe": False, 
                "message": "КРИТИЧЕСКАЯ ОШИБКА! Конструкция потеряет устойчивость!"
            }
        
        # Проверяем перегрузки
        max_load = max([r['load_ratio'] for r in result['data']])
        if max_load > 1.0:
             return {
                "safe": False, 
                "message": f"ОПАСНО! Балка перегрузится до {int(max_load*100)}%"
            }
            
        return {"safe": True, "message": "Безопасно. Макс нагрузка перераспределится."}