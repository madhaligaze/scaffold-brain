# modules/dynamics.py
"""
Модуль динамических нагрузок для анализа ветра, вибрации и резонанса.
Критично для высотных лесов и конструкций около работающего оборудования.
"""
import math
from typing import List, Dict

class DynamicLoadAnalyzer:
    """
    Анализатор динамических нагрузок:
    - Ветровая нагрузка (по высоте и открытости)
    - Вибрационная нагрузка (от оборудования)
    - Резонансный анализ
    """
    
    # Константы для ветровой нагрузки (упрощенная модель СНиП)
    BASE_WIND_PRESSURE = 300  # Па (базовое давление ветра для средней полосы)
    HEIGHT_FACTOR_COEFF = 0.15  # Коэффициент увеличения с высотой
    
    # Параметры для вибрации
    CONVEYOR_VIBRATION_FREQ = 25  # Гц (типичная частота конвейера)
    MACHINE_VIBRATION_AMPLITUDE = 0.002  # м (2 мм)
    
    def __init__(self):
        pass
    
    def calculate_wind_load(self, nodes: List[Dict], beams: List[Dict], 
                           wind_speed: float = 20.0, wind_direction: str = "X") -> Dict:
        """
        Рассчитывает ветровую нагрузку на конструкцию.
        
        Args:
            nodes: список узлов
            beams: список балок
            wind_speed: скорость ветра в м/с (по умолчанию 20 м/с ≈ 72 км/ч)
            wind_direction: направление ветра ("X", "Y", или "XY" - угол 45°)
        
        Returns:
            {
                "status": "OK"|"WARNING"|"DANGER",
                "total_force_N": float,
                "max_pressure_Pa": float,
                "critical_beams": [...],
                "recommendations": [...]
            }
        """
        # 1. Динамическое давление ветра: q = 0.5 * ρ * v²
        air_density = 1.225  # кг/м³
        dynamic_pressure = 0.5 * air_density * (wind_speed ** 2)
        
        # 2. Находим максимальную высоту конструкции
        max_height = max([n['z'] for n in nodes]) if nodes else 0
        
        # 3. Коэффициент высоты (ветер усиливается с высотой)
        height_factor = 1.0 + (max_height * self.HEIGHT_FACTOR_COEFF)
        
        # 4. Рассчитываем эффективную площадь, обдуваемую ветром
        exposed_area = self._calculate_exposed_area(nodes, beams, wind_direction)
        
        # 5. Суммарная ветровая сила
        total_wind_force = dynamic_pressure * height_factor * exposed_area
        
        # 6. Находим критичные балки (высокие и перпендикулярные ветру)
        critical_beams = []
        
        for beam in beams:
            start_node = self._find_node(beam['start'], nodes)
            end_node = self._find_node(beam['end'], nodes)
            
            if not start_node or not end_node:
                continue
            
            avg_height = (start_node['z'] + end_node['z']) / 2
            
            # Проверяем ориентацию балки относительно ветра
            if self._is_perpendicular_to_wind(start_node, end_node, wind_direction):
                beam_length = self._beam_length(start_node, end_node)
                beam_height_factor = 1.0 + (avg_height * self.HEIGHT_FACTOR_COEFF)
                
                # Давление на эту балку
                beam_pressure = dynamic_pressure * beam_height_factor
                
                # Сила на балку (упрощенно: давление * длина * диаметр трубы)
                pipe_diameter = 0.048  # м (труба 48мм)
                beam_wind_force = beam_pressure * beam_length * pipe_diameter
                
                if beam_wind_force > 100:  # Порог 100 Н
                    critical_beams.append({
                        "id": beam['id'],
                        "force_N": round(beam_wind_force, 1),
                        "height_m": round(avg_height, 2),
                        "pressure_Pa": round(beam_pressure, 1)
                    })
        
        # 7. Статус и рекомендации
        status = "OK"
        recommendations = []
        
        if total_wind_force > 5000:  # 5 кН
            status = "DANGER"
            recommendations.append("🌪️ ОПАСНО! Ветровая нагрузка критична. Требуются дополнительные растяжки.")
        elif total_wind_force > 2000:  # 2 кН
            status = "WARNING"
            recommendations.append("⚠️ Высокая ветровая нагрузка. Рекомендуются анкерные крепления каждые 3 метра.")
        else:
            recommendations.append("✓ Ветровая нагрузка в пределах нормы.")
        
        if max_height > 10:
            recommendations.append(f"⚠️ Высота {max_height:.1f}м требует усиленных креплений к зданию.")
        
        return {
            "status": status,
            "wind_speed_ms": wind_speed,
            "total_force_N": round(total_wind_force, 1),
            "max_pressure_Pa": round(dynamic_pressure * height_factor, 1),
            "max_height_m": round(max_height, 2),
            "exposed_area_m2": round(exposed_area, 2),
            "critical_beams": sorted(critical_beams, key=lambda x: x['force_N'], reverse=True)[:10],
            "recommendations": recommendations
        }
    
    def calculate_vibration_impact(self, nodes: List[Dict], beams: List[Dict],
                                   vibration_source: Dict) -> Dict:
        """
        Анализирует воздействие вибрации от оборудования (конвейер, станок).
        
        Args:
            nodes: список узлов
            beams: список балок
            vibration_source: {
                "x": float, "y": float, "z": float,  # координаты источника
                "frequency_hz": float,  # частота (Гц)
                "amplitude_m": float,  # амплитуда (метры)
                "type": "conveyor"|"machine"|"hammer"
            }
        
        Returns:
            {
                "status": "OK"|"WARNING"|"RESONANCE_DANGER",
                "natural_frequency_hz": float,
                "forced_frequency_hz": float,
                "resonance_risk": float (0-1),
                "recommendations": [...]
            }
        """
        # 1. Рассчитываем собственную частоту конструкции (упрощенная модель)
        natural_freq = self._estimate_natural_frequency(nodes, beams)
        
        # 2. Частота вынужденных колебаний (от оборудования)
        forced_freq = vibration_source.get('frequency_hz', self.CONVEYOR_VIBRATION_FREQ)
        
        # 3. Расстояние до источника вибрации
        distances_to_source = []
        for node in nodes:
            dist = math.sqrt(
                (node['x'] - vibration_source['x'])**2 +
                (node['y'] - vibration_source['y'])**2 +
                (node['z'] - vibration_source['z'])**2
            )
            distances_to_source.append(dist)
        
        min_distance = min(distances_to_source) if distances_to_source else float('inf')
        
        # 4. Проверка резонанса (опасно, если частоты близки)
        freq_ratio = forced_freq / natural_freq if natural_freq > 0 else 0
        
        # Резонанс возникает при freq_ratio ≈ 1.0 (±10%)
        resonance_risk = 0
        if 0.9 <= freq_ratio <= 1.1:
            resonance_risk = 1.0 - abs(1.0 - freq_ratio) / 0.1
        elif 0.8 <= freq_ratio <= 1.2:
            resonance_risk = 0.5 - abs(1.0 - freq_ratio) / 0.4
        
        # 5. Затухание вибрации с расстоянием
        attenuation_factor = 1.0 / (1.0 + min_distance / 2.0)
        
        # 6. Эффективная амплитуда вибрации на конструкции
        source_amplitude = vibration_source.get('amplitude_m', self.MACHINE_VIBRATION_AMPLITUDE)
        effective_amplitude = source_amplitude * attenuation_factor
        
        # 7. Статус и рекомендации
        status = "OK"
        recommendations = []
        
        if resonance_risk > 0.7:
            status = "RESONANCE_DANGER"
            recommendations.append(
                f"🔴 РЕЗОНАНС! Частота конструкции ({natural_freq:.1f} Гц) совпадает с "
                f"частотой оборудования ({forced_freq:.1f} Гц). Требуются виброгасители!"
            )
        elif resonance_risk > 0.3:
            status = "WARNING"
            recommendations.append(
                f"⚠️ Риск резонанса {int(resonance_risk*100)}%. Рекомендуется изменить жесткость конструкции."
            )
        else:
            recommendations.append("✓ Резонанс маловероятен. Частоты разнесены достаточно.")
        
        if min_distance < 1.0:
            recommendations.append(
                f"⚠️ Источник вибрации очень близко ({min_distance:.2f}м). "
                f"Используйте виброизолирующие прокладки."
            )
        
        if effective_amplitude > 0.005:  # 5 мм
            recommendations.append(
                f"⚠️ Амплитуда вибрации на конструкции: {effective_amplitude*1000:.1f}мм. "
                f"Превышает безопасный предел (5 мм)."
            )
        
        return {
            "status": status,
            "natural_frequency_hz": round(natural_freq, 2),
            "forced_frequency_hz": round(forced_freq, 2),
            "resonance_risk": round(resonance_risk, 3),
            "frequency_ratio": round(freq_ratio, 3),
            "min_distance_to_source_m": round(min_distance, 2),
            "effective_amplitude_mm": round(effective_amplitude * 1000, 2),
            "recommendations": recommendations
        }
    
    def suggest_vibration_dampening(self, analysis_result: Dict) -> List[Dict]:
        """
        Предлагает меры по снижению вибрации.
        
        Returns:
            Список рекомендаций с конкретными решениями
        """
        solutions = []
        
        resonance_risk = analysis_result.get('resonance_risk', 0)
        
        if resonance_risk > 0.7:
            # Критичный резонанс
            solutions.append({
                "priority": "CRITICAL",
                "solution": "Установка демпферов",
                "description": "Виброгасящие элементы (резиновые прокладки) между лесами и конвейером",
                "cost_estimate": "Средняя",
                "effectiveness": "90%"
            })
            
            solutions.append({
                "priority": "CRITICAL",
                "solution": "Изменение жесткости",
                "description": "Добавить/удалить диагональные связи для сдвига собственной частоты",
                "cost_estimate": "Низкая",
                "effectiveness": "70%"
            })
        
        elif resonance_risk > 0.3:
            # Умеренный риск
            solutions.append({
                "priority": "HIGH",
                "solution": "Усиление конструкции",
                "description": "Дополнительные горизонтальные связи для повышения жесткости",
                "cost_estimate": "Низкая",
                "effectiveness": "60%"
            })
        
        # Если близко к источнику
        if analysis_result.get('min_distance_to_source_m', 10) < 2.0:
            solutions.append({
                "priority": "MEDIUM",
                "solution": "Виброизоляция основания",
                "description": "Резиновые маты под опорные узлы лесов",
                "cost_estimate": "Низкая",
                "effectiveness": "50%"
            })
        
        return solutions
    
    # === ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ===
    
    def _find_node(self, node_id: str, nodes: List[Dict]) -> Dict:
        """Поиск узла по ID"""
        for n in nodes:
            if n['id'] == node_id:
                return n
        return None
    
    def _beam_length(self, node1: Dict, node2: Dict) -> float:
        """Длина балки"""
        return math.sqrt(
            (node1['x'] - node2['x'])**2 +
            (node1['y'] - node2['y'])**2 +
            (node1['z'] - node2['z'])**2
        )
    
    def _calculate_exposed_area(self, nodes: List[Dict], beams: List[Dict], 
                               wind_direction: str) -> float:
        """
        Рассчитывает эффективную площадь, обдуваемую ветром.
        Упрощенный подход: проецируем балки на плоскость, перпендикулярную ветру.
        """
        total_area = 0
        pipe_diameter = 0.048  # м
        
        for beam in beams:
            start_node = self._find_node(beam['start'], nodes)
            end_node = self._find_node(beam['end'], nodes)
            
            if not start_node or not end_node:
                continue
            
            beam_len = self._beam_length(start_node, end_node)
            
            # Площадь = длина * диаметр (цилиндрическая балка)
            total_area += beam_len * pipe_diameter
        
        # Корректировка на направление (упрощенно)
        if wind_direction == "XY":
            total_area *= 0.707  # cos(45°)
        
        return total_area
    
    def _is_perpendicular_to_wind(self, node1: Dict, node2: Dict, wind_direction: str) -> bool:
        """Проверяет, перпендикулярна ли балка направлению ветра"""
        dx = abs(node1['x'] - node2['x'])
        dy = abs(node1['y'] - node2['y'])
        
        if wind_direction == "X":
            return dy > dx  # Балка идет в направлении Y
        elif wind_direction == "Y":
            return dx > dy  # Балка идет в направлении X
        else:  # "XY"
            return True  # Все балки частично перпендикулярны
    
    def _estimate_natural_frequency(self, nodes: List[Dict], beams: List[Dict]) -> float:
        """
        Оценка собственной частоты конструкции (упрощенная модель).
        
        Формула: f ≈ (1 / 2π) * sqrt(k / m)
        где k - жесткость, m - масса
        """
        if not nodes or not beams:
            return 0
        
        # 1. Оценка массы
        # Труба 48x3 весит ~3.5 кг/м
        total_mass = 0
        for beam in beams:
            start_node = self._find_node(beam['start'], nodes)
            end_node = self._find_node(beam['end'], nodes)
            
            if start_node and end_node:
                beam_len = self._beam_length(start_node, end_node)
                total_mass += beam_len * 3.5  # кг
        
        # 2. Оценка жесткости (упрощенно через модуль упругости и геометрию)
        # E = 2.1e11 Па, inertia = 1.1e-7 м4 (момент инерции трубы 48х3)
        E = 2.1e11
        inertia = 1.1e-7
        
        # Средняя длина балки
        avg_beam_length = 2.0  # м (примерно)
        
        # Жесткость балки: k ≈ 3*E*I / L³
        k = 3 * E * inertia / (avg_beam_length ** 3)
        
        # Собственная частота
        omega = math.sqrt(k / total_mass) if total_mass > 0 else 0
        frequency = omega / (2 * math.pi)
        
        return frequency


class ProgressiveCollapseAnalyzer:
    """
    Анализатор прогрессирующего обрушения.
    Проверяет: если одна балка сломается, упадет ли вся конструкция?
    """
    
    def __init__(self, physics_engine):
        self.physics = physics_engine
    
    def analyze_progressive_collapse(self, nodes: List[Dict], beams: List[Dict],
                                    failure_scenario: str = "random") -> Dict:
        """
        Симулирует отказ критичных элементов и проверяет каскадное обрушение.
        
        Args:
            nodes: список узлов
            beams: список балок
            failure_scenario: "random", "impact", "overload"
        
        Returns:
            {
                "status": "SAFE"|"VULNERABLE"|"CRITICAL",
                "critical_elements": [...],  # Элементы, чей отказ приведет к обрушению
                "cascade_risk": float (0-1),
                "recommendations": [...]
            }
        """
        critical_elements = []
        
        # 1. Тестируем удаление каждой балки по очереди
        for beam in beams:
            # Симулируем удаление
            result = self.physics.simulate_removal(nodes, beams, beam['id'])
            
            if not result["safe"]:
                # Эта балка критична!
                critical_elements.append({
                    "id": beam['id'],
                    "criticality": "HIGH",
                    "failure_consequence": result["message"]
                })
        
        # 2. Рассчитываем риск каскадного обрушения
        total_beams = len(beams)
        critical_count = len(critical_elements)
        
        cascade_risk = critical_count / total_beams if total_beams > 0 else 0
        
        # 3. Статус и рекомендации
        status = "SAFE"
        recommendations = []
        
        if cascade_risk > 0.3:
            status = "CRITICAL"
            recommendations.append(
                f"🔴 КРИТИЧНО! {critical_count} из {total_beams} балок критичны. "
                f"Отказ одной приведет к обрушению."
            )
            recommendations.append("Требуется резервирование: дублирование опорных элементов.")
        elif cascade_risk > 0.1:
            status = "VULNERABLE"
            recommendations.append(
                f"⚠️ {critical_count} критичных элементов. Рекомендуется усиление."
            )
        else:
            recommendations.append("✓ Конструкция устойчива к отказу отдельных элементов.")
        
        return {
            "status": status,
            "total_beams": total_beams,
            "critical_count": critical_count,
            "cascade_risk": round(cascade_risk, 3),
            "critical_elements": critical_elements[:10],  # Топ-10 критичных
            "recommendations": recommendations
        }
