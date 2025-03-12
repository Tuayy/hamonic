import os
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import random
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import threading
from datetime import datetime

# กำหนด Harmonic Patterns เริ่มต้น
HARMONIC_PATTERNS_BASE = {
    "Gartley": {"AB/XA": [0.618, 0.618], "BC/AB": [0.382, 0.886], "CD/BC": [1.618, 2.618]},
    "Bat": {"AB/XA": [0.382, 0.5], "BC/AB": [0.382, 0.886], "CD/BC": [1.618, 2.618]},
    "Butterfly": {"AB/XA": [0.786, 0.786], "BC/AB": [0.382, 0.886], "CD/BC": [1.618, 2.618]},
    "Crab": {"AB/XA": [0.382, 0.618], "BC/AB": [0.382, 0.886], "CD/BC": [2.618, 3.618]},
    "Deep Crab": {"AB/XA": [0.886, 0.886], "BC/AB": [0.382, 0.886], "CD/BC": [2.24, 3.618]},
    "Shark": {"AB/XA": [0.886, 1.13], "BC/AB": [1.13, 1.618], "CD/BC": [1.618, 2.24]},
    "Cypher": {"AB/XA": [0.382, 0.618], "BC/AB": [1.13, 1.414], "CD/BC": [1.618, 2.24]}
}

# ฟังก์ชันตรวจสอบ pattern
def check_harmonic_ratios(xa, ab, bc, cd, pattern_params, tolerance):
    ratios = {}
    if xa == 0 or ab == 0 or bc == 0 or cd == 0:
        return False, ratios
    try:
        ratios["AB/XA"] = ab / xa
        ratios["BC/AB"] = bc / ab
        ratios["CD/BC"] = cd / bc
    except ZeroDivisionError:
        return False, ratios
    
    for key, (min_val, max_val) in pattern_params.items():
        adjusted_min = min_val * (1 - tolerance)
        adjusted_max = max_val * (1 + tolerance)
        if not (adjusted_min <= ratios[key] <= adjusted_max):
            return False, ratios
    return True, ratios

# โหลดข้อมูลจาก CSV
def load_price_data(csv_file, date_column="DateTime", price_column="Close"):
    try:
        df = pd.read_csv(csv_file)
        if date_column not in df.columns or price_column not in df.columns:
            raise ValueError(f"คอลัมน์ที่ต้องการไม่พบ คอลัมน์ที่มี: {df.columns}")
        
        # ตรวจสอบว่าคอลัมน์ราคาเป็นตัวเลขหรือไม่
        if not pd.api.types.is_numeric_dtype(df[price_column]):
            raise ValueError(f"คอลัมน์ {price_column} ต้องเป็นตัวเลข")
        
        # แปลงวันที่เป็น DatetimeIndex และจัดการ NaN
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df = df.dropna(subset=[date_column, price_column])  # ลบแถวที่มี NaN ในทั้งวันที่และราคา
        df.set_index(date_column, inplace=True)  # ตั้งค่าเป็น DatetimeIndex
        return df[price_column]
    except Exception as e:
        raise ValueError(f"ไม่สามารถโหลดข้อมูลได้: {str(e)}")

# Resample ข้อมูล
def resample_data(prices, timeframe):
    if prices.empty or prices.isna().all():
        raise ValueError("ข้อมูลราคาว่างเปล่าหรือมีค่า NaN เท่านั้น")
    timeframe_mapping = {
        "5M": "5T",  # 5 นาที (ใช้ 'T' สำหรับนาที)
        "15M": "15T",  # 15 นาที
        "30M": "30T",  # 30 นาที
        "1H": "1H",  # 1 ชั่วโมง
        "4H": "4H",  # 4 ชั่วโมง
        "1D": "1D",  # 1 วัน
        "1W": "1W",  # 1 สัปดาห์
        "1M": "1M"   # 1 เดือน
    }
    if timeframe not in timeframe_mapping:
        raise ValueError(f"ไม่รองรับ timeframe: {timeframe}")
    resampled = prices.resample(timeframe_mapping[timeframe]).last().dropna()
    if resampled.empty:
        raise ValueError(f"ไม่มีข้อมูลหลังจาก resample เป็น {timeframe}")
    return resampled

# หา pivot points
def find_pivots(data, order=5):
    highs = argrelextrema(data.values, np.greater, order=order)[0]
    lows = argrelextrema(data.values, np.less, order=order)[0]
    if len(highs) + len(lows) < 5:
        raise ValueError(f"พบ pivot ไม่เพียงพอ ({len(highs) + len(lows)} จุด) ลองลด order")
    return highs, lows

# คำนวณ Take Profit
def calculate_take_profits(x, a, d, is_bullish):
    """คำนวณ TP1 และ TP2 จากจุด X, A, และ D"""
    if not all(isinstance(v, (int, float)) and not np.isnan(v) for v in [x, a, d]):
        raise ValueError("ข้อมูลราคาไม่ถูกต้องหรือเป็น NaN")
    
    xa_range = abs(a - x)
    if xa_range <= 0:
        raise ValueError("ระยะ XA ต้องมากกว่า 0")
    
    tp1_offset = xa_range * 0.382  # Fibonacci 0.382
    tp2_offset = xa_range * 0.618  # Fibonacci 0.618
    
    if is_bullish:  # Bullish Pattern: TP อยู่สูงกว่าจุด D
        tp1 = d + tp1_offset
        tp2 = d + tp2_offset
    else:  # Bearish Pattern: TP อยู่ต่ำกว่าจุด D
        tp1 = d - tp1_offset
        tp2 = d - tp2_offset
    
    return tp1, tp2

# Backtest Harmonic Patterns
def backtest_harmonics(prices, timeframe, individual):
    try:
        resampled_prices = resample_data(prices, timeframe)
        highs, lows = find_pivots(resampled_prices)
        pivots = sorted([(i, resampled_prices.iloc[i], "H" if i in highs else "L") for i in list(highs) + list(lows)])
        
        if len(pivots) < 5:
            return 0, []
        
        patterns_found = []
        pattern_params = individual["params"]
        tolerance = individual["tolerance"]
        
        # ใช้ set เพื่อเก็บตำแหน่ง pivot และลำดับที่ใช้แล้ว เพื่อป้องกันการนับซ้ำ
        used_pivots = set()
        
        for i in range(len(pivots) - 4):
            x_idx, x_price = pivots[i][0], pivots[i][1]
            a_idx, a_price = pivots[i+1][0], pivots[i+1][1]
            b_idx, b_price = pivots[i+2][0], pivots[i+2][1]
            c_idx, c_price = pivots[i+3][0], pivots[i+3][1]
            d_idx, d_price = pivots[i+4][0], pivots[i+4][1]
            
            # ตรวจสอบว่าพivot นี้ถูกใช้แล้วหรือไม่
            if any(idx in used_pivots for idx in [x_idx, a_idx, b_idx, c_idx, d_idx]):
                continue
            
            # ตรวจสอบว่าค่าทั้งหมดเป็นตัวเลขหรือไม่
            if not all(isinstance(p, (int, float)) and not np.isnan(p) for p in [x_price, a_price, b_price, c_price, d_price]):
                continue
            
            xa = abs(a_price - x_price)
            ab = abs(b_price - a_price)
            bc = abs(c_price - b_price)
            cd = abs(d_price - c_price)
            
            # ตรวจสอบทิศทาง (Bullish ถ้า X < A)
            is_bullish = x_price < a_price
            
            for pattern_type, params in pattern_params.items():
                is_valid, _ = check_harmonic_ratios(xa, ab, bc, cd, params, tolerance)
                if is_valid:
                    try:
                        tp1, tp2 = calculate_take_profits(x_price, a_price, d_price, is_bullish)
                        # ตรวจสอบว่า TP มีค่าและไม่เป็น NaN
                        if not (np.isnan(tp1) or np.isnan(tp2)):
                            patterns_found.append({
                                "type": pattern_type,
                                "TP1": np.round(tp1, 4),  # ปัดทศนิยมเพื่อความสม่ำเสมอ
                                "TP2": np.round(tp2, 4)   # ปัดทศนิยมเพื่อความสม่ำเสมอ
                            })
                            # เพิ่ม pivot ที่ใช้แล้วลงใน set เพื่อป้องกันการนับซ้ำ
                            used_pivots.update([x_idx, a_idx, b_idx, c_idx, d_idx])
                    except Exception as e:
                        print(f"ข้อผิดพลาดในการคำนวณ TP สำหรับ {pattern_type}: {str(e)}")
                        continue
        
        # กรองแพทเทิร์นที่ซ้ำกัน (ตามประเภทและค่า TP)
        unique_patterns = []
        seen_patterns = set()
        for pattern in patterns_found:
            pattern_key = (pattern["type"], pattern["TP1"], pattern["TP2"])
            if pattern_key not in seen_patterns:
                seen_patterns.add(pattern_key)
                unique_patterns.append(pattern)
        
        # ดีบักเพื่อตรวจสอบจำนวนและรายละเอียดแพทเทิร์น
        print(f"จำนวนแพทเทิร์นทั้งหมด (patterns_found): {len(patterns_found)}")
        print(f"จำนวนแพทเทิร์นที่ไม่ซ้ำ (unique_patterns): {len(unique_patterns)}")
        print(f"รายละเอียดแพทเทิร์น: {[p['type'] for p in unique_patterns]}")
        
        return len(unique_patterns), unique_patterns
    except Exception as e:
        print(f"ข้อผิดพลาดใน backtest_harmonics: {str(e)}")
        return 0, []

# Genetic Algorithm Functions
def create_individual(pattern_base):
    individual = {"params": {}, "tolerance": 0.1}  # ตั้งค่า tolerance เป็นค่าคงที่
    for pattern, ratios in pattern_base.items():
        individual["params"][pattern] = {
            "AB/XA": sorted([random.uniform(ratios["AB/XA"][0] * 0.9, ratios["AB/XA"][1] * 1.1),  # ลดช่วงการสุ่ม
                            random.uniform(ratios["AB/XA"][0] * 0.9, ratios["AB/XA"][1] * 1.1)]),
            "BC/AB": sorted([random.uniform(ratios["BC/AB"][0] * 0.9, ratios["BC/AB"][1] * 1.1),
                            random.uniform(ratios["BC/AB"][0] * 0.9, ratios["BC/AB"][1] * 1.1)]),
            "CD/BC": sorted([random.uniform(ratios["CD/BC"][0] * 0.9, ratios["CD/BC"][1] * 1.1),
                            random.uniform(ratios["CD/BC"][0] * 0.9, ratios["CD/BC"][1] * 1.1)])
        }
    return individual

def crossover(parent1, parent2):
    child = {"params": {}, "tolerance": 0.1}  # คง tolerance เป็นค่าคงที่ 0.1
    for pattern in parent1["params"]:
        child["params"][pattern] = {}
        for key in parent1["params"][pattern]:
            if random.random() < 0.3:  # ลดโอกาสในการเลือกจาก parent2
                child["params"][pattern][key] = parent1["params"][pattern][key]
            else:
                child["params"][pattern][key] = parent2["params"][pattern][key]
    return child

def mutate(individual, mutation_rate=0.02):  # ลด mutation_rate ลงอีก
    for pattern in individual["params"]:
        for key in individual["params"][pattern]:
            if random.random() < mutation_rate:
                individual["params"][pattern][key][0] *= random.uniform(0.98, 1.02)  # ลดช่วงการเปลี่ยนแปลงมากขึ้น
                individual["params"][pattern][key][1] *= random.uniform(0.98, 1.02)  # ลดช่วงการเปลี่ยนแปลงมากขึ้น
    return individual

def genetic_algorithm(prices, timeframe, population_size=20, generations=10, output_text=None):
    population = [create_individual(HARMONIC_PATTERNS_BASE) for _ in range(population_size)]
    best_patterns = []
    fitness_history = []  # เก็บ best_fitness เพื่อ smoothing
    
    for gen in range(generations):
        # ลดการใช้ทรัพยากรโดยตรวจสอบข้อมูลก่อน
        if prices.empty:
            if output_text:
                output_text.insert(tk.END, "ข้อมูลราคาว่างเปล่า กรุณาตรวจสอบไฟล์ CSV\n")
            return None, []
        
        fitness_scores = []
        for ind in population:
            count, patterns = backtest_harmonics(prices, timeframe, ind)
            fitness_scores.append((ind, count, patterns))
        
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        survivors = [ind for ind, _, _ in fitness_scores[:population_size // 5]]  # เก็บเฉพาะ top 20%
        next_gen = survivors.copy()
        while len(next_gen) < population_size:
            parent1, parent2 = random.sample(survivors, 2)
            child = crossover(parent1, parent2)
            child = mutate(child)
            next_gen.append(child)
        
        population = next_gen
        best_individual, best_fitness, best_patterns = fitness_scores[0]
        fitness_history.append(best_fitness)
        smoothed_fitness = sum(fitness_history[-10:]) / min(10, len(fitness_history))  # เพิ่ม smoothing เป็น 10 รอบล่าสุด
        
        if output_text:
            output_text.insert(tk.END, f"รอบที่ {gen + 1}/{generations} - คะแนนดีที่สุด: {smoothed_fitness:.1f}\n")
            output_text.see(tk.END)
    
    return best_individual, best_patterns

# GUI Class
class HarmonicBacktestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("การทดสอบย้อนหลังแพทเทิร์นฮาร์โมนิกด้วย GA")
        self.root.geometry("600x500")

        tk.Label(root, text="เลือกไฟล์ CSV:").grid(row=0, column=0, padx=5, pady=5)
        self.file_path = tk.StringVar()
        tk.Entry(root, textvariable=self.file_path, width=40).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(root, text="เรียกดู", command=self.browse_file).grid(row=0, column=2, padx=5, pady=5)

        tk.Label(root, text="เลือกช่วงเวลา:").grid(row=1, column=0, padx=5, pady=5)
        # ตัวเลือก Timeframe เดิม + เพิ่มใหม่
        self.timeframes = ttk.Combobox(root, values=["5M", "15M", "30M", "1H", "4H", "1D", "1W", "1M"], state="readonly")
        self.timeframes.grid(row=1, column=1, padx=5, pady=5)
        self.timeframes.set("1D")

        tk.Label(root, text="ขนาดประชากร:").grid(row=2, column=0, padx=5, pady=5)
        self.pop_size = tk.Entry(root)
        self.pop_size.grid(row=2, column=1, padx=5, pady=5)
        self.pop_size.insert(0, "20")

        tk.Label(root, text="จำนวนรอบ:").grid(row=3, column=0, padx=5, pady=5)
        self.generations = tk.Entry(root)
        self.generations.grid(row=3, column=1, padx=5, pady=5)
        self.generations.insert(0, "10")

        tk.Button(root, text="เริ่มทดสอบ", command=self.run_backtest_thread).grid(row=4, column=1, pady=10)

        self.output_text = scrolledtext.ScrolledText(root, width=70, height=20)
        self.output_text.grid(row=5, column=0, columnspan=3, padx=5, pady=5)

    def browse_file(self):
        file = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file:
            self.file_path.set(file)

    def run_backtest(self):
        csv_file = self.file_path.get()
        if not csv_file or not os.path.exists(csv_file):
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "ไฟล์ CSV ไม่ถูกต้องหรือไม่พบ\n")
            return
    
        try:
            prices = load_price_data(csv_file)
            if prices.empty:
                self.output_text.insert(tk.END, "ข้อมูลราคาว่างเปล่า กรุณาตรวจสอบไฟล์ CSV\n")
                return
            
            timeframe = self.timeframes.get()
            pop_size = int(self.pop_size.get()) if self.pop_size.get().isdigit() else 20
            generations = int(self.generations.get()) if self.generations.get().isdigit() else 10
            
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, f"กำลังวิเคราะห์และปรับแต่งพารามิเตอร์สำหรับ {timeframe}...\n")
            best_individual, patterns_found = genetic_algorithm(prices, timeframe, pop_size, generations, self.output_text)
            
            if best_individual is None or not patterns_found:
                self.output_text.insert(tk.END, "ไม่พบแพทเทิร์นฮาร์โมนิก กรุณาตรวจสอบข้อมูลหรือพารามิเตอร์\n")
                self.output_text.see(tk.END)
                return
            
            self.output_text.insert(tk.END, f"\nการวิเคราะห์เสร็จสิ้น!\n")
            self.output_text.insert(tk.END, "พารามิเตอร์ที่ดีที่สุดสำหรับแต่ละแพทเทิร์น:\n")
            
            # จัดเรียงข้อมูลพารามิเตอร์และ TP ตามประเภทแพทเทิร์น
            result_data = {"Pattern": [], "Ratio": [], "Min Value": [], "Max Value": [], "TP1": [], "TP2": []}
            pattern_tps = {}  # เก็บ TP สำหรับแต่ละแพทเทิร์น
            
            # เก็บพารามิเตอร์
            for pattern, ratios in best_individual["params"].items():
                self.output_text.insert(tk.END, f"- {pattern}:\n")
                for ratio_key, (min_val, max_val) in ratios.items():
                    self.output_text.insert(tk.END, f"  {ratio_key}: {min_val:.4f} ถึง {max_val:.4f}\n")
                    result_data["Pattern"].append(pattern)
                    result_data["Ratio"].append(ratio_key)
                    result_data["Min Value"].append(min_val)
                    result_data["Max Value"].append(max_val)
                    result_data["TP1"].append("")
                    result_data["TP2"].append("")
            
            # เก็บ TP และแมปกับประเภทแพทเทิร์น
            for pattern in patterns_found:
                pattern_tps[pattern["type"]] = (pattern["TP1"], pattern["TP2"])
            
            # เพิ่ม TP เข้ากับข้อมูลตามประเภทแพทเทิร์น (หลังจาก CD/BC)
            for i in range(len(result_data["Pattern"])):
                pattern = result_data["Pattern"][i]
                if pattern in pattern_tps and result_data["Ratio"][i] == "CD/BC":
                    result_data["TP1"][i] = pattern_tps[pattern][0]
                    result_data["TP2"][i] = pattern_tps[pattern][1]
            
            # ดีบักเพื่อตรวจสอบจำนวนแพทเทิร์น
            print(f"จำนวนแพทเทิร์นที่พบ (patterns_found): {len(patterns_found)}")
            print(f"จำนวนแพทเทิร์นใน result_data: {len([p for p in result_data['Pattern'] if p in pattern_tps])}")
            print(f"รายละเอียดแพทเทิร์น: {[p['type'] for p in patterns_found]}")
            
            # บันทึกผลลัพธ์เป็น CSV (ไม่รวม Summary)
            result_df = pd.DataFrame(result_data)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"harmonic_results_{timeframe}_{timestamp}.csv"
            result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            self.output_text.insert(tk.END, f"\nบันทึกผลลัพธ์ลงไฟล์: {output_file}\n")
            self.output_text.see(tk.END)
            
        except Exception as e:
            self.output_text.insert(tk.END, f"เกิดข้อผิดพลาด: {str(e)}\n")
            self.output_text.see(tk.END)

    def run_backtest_thread(self):
        thread = threading.Thread(target=self.run_backtest)
        thread.start()

# รัน GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = HarmonicBacktestApp(root)
    # root.iconbitmap("hamo.ico")  # ถ้าไม่มีไฟล์ hamo.ico ให้คอมเมนต์บรรทัดนี้
    root.mainloop()