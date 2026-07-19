"""
generate_save_steps.py

train_and_save.py に渡す save_steps リストを生成する。
対数間隔で num_points 個生成し、重複除去・total_steps以下に絞る。
"""
import numpy as np

total_steps = 78000
num_points = 700

raw = np.logspace(0, np.log10(total_steps), num_points)
steps = sorted(set(np.round(raw).astype(int).tolist()))
steps = [s for s in steps if 0 <= s <= total_steps]

print(f"Total unique steps: {len(steps)}")
print()
# train_and_save.py にそのまま貼れる形式で出力
print("save_steps = [")
for i in range(0, len(steps), 15):
    chunk = steps[i:i+15]
    print("    " + ", ".join(str(s) for s in chunk) + ",")
print("]")
print()

# コマンドライン用のカンマ区切り文字列(Sacred用)
cmdline_str = "[" + ",".join(str(s) for s in steps) + "]"
print("Command-line format (for Sacred 'with' argument):")
print(f"'save_steps={cmdline_str}'")

# ファイルにも保存しておく(コピペミス防止)
with open("save_steps_dense.txt", "w") as f:
    f.write(cmdline_str)
print("\nSaved to save_steps_dense.txt")