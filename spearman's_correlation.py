import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

sns.set(style='whitegrid', palette='deep', font_scale=1.2)


np.random.seed(0)
students_grade = np.random.randint(50, 100, 30)  # 50부터 100 사이의 성적
students_confidence = np.sort(np.random.randint(1, 10, 30))[::-1]  # 높은 순위부터 낮은 순위로 자신감 수준

corr_coef1, _ = spearmanr(students_grade, students_confidence)
print(f"학생들의 성적과 자신감 수준 사이의 스피어만 상관계수: {corr_coef1:.2f}")

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=students_grade, y=students_confidence, color='blue', s=100)
plt.title(f"Grade vs. Confident (ρ = {corr_coef1:.2f})")
plt.xlabel("Grade")
plt.ylabel("Confident")

ages = np.arange(18, 60)  # 18세부터 59세까지
smartphone_usage = np.sort(np.random.randint(1, 5, len(ages)) * -1 + np.max(ages))[::-1]  # 사용 시간 감소

corr_coef2, _ = spearmanr(ages, smartphone_usage)
print(f"연령과 스마트폰 사용 시간 사이의 스피어만 상관계수: {corr_coef2:.2f}")

plt.subplot(1, 2, 2)
sns.scatterplot(x=ages, y=smartphone_usage, color='green', s=100)
plt.title(f"Age vs. Phone use time (ρ = {corr_coef2:.2f})")
plt.xlabel("Age")
plt.ylabel("Phone use time")

plt.tight_layout()
plt.show()


