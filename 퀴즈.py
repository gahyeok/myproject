
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# [1] 데이터 불러오기 및 전처리
data = pd.read_excel("서울대기오염_2019.xlsx")
data = data[data['날짜'] != '전체']  # 요약 행 제거

# 컬럼명 정리
data = data.rename(columns={
    '날짜': 'date',
    '측정소명': 'district',
    '미세먼지': 'pm10',
    '초미세먼지': 'pm25'
})
data = data[['date', 'district', 'pm10', 'pm25']]

# 자료형 변환
data['date'] = pd.to_datetime(data['date'])
data['pm10'] = data['pm10'].astype(float)
data['pm25'] = data['pm25'].astype(float)

# [2] 파생변수 생성
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

def get_season(month):
    if month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'
    else:
        return 'winter'

data['season'] = data['month'].apply(get_season)

# [3] 전처리 완료 파일 저장
data.to_csv("card_output.csv", index=False)

# [4] 분석 및 통계 계산

# 4-1. 전체 PM10 평균
avg_pm10 = data['pm10'].mean()

# 5-1. PM10 최댓값 발생 날짜 및 지역
max_pm10_info = data[data['pm10'] == data['pm10'].max()][['date', 'district', 'pm10']]

# 6. 구별 평균 PM10 상위 5개
avg_pm10_by_gu = data.groupby('district')['pm10'].mean().reset_index().sort_values(by='pm10', ascending=False).head(5)

# 7. 계절별 PM10, PM2.5 평균
avg_by_season = data.groupby('season')[['pm10', 'pm25']].mean().reset_index().sort_values(by='pm10')

# 8. PM10 등급화
def classify_pm10(value):
    if value <= 30:
        return 'good'
    elif value <= 80:
        return 'normal'
    elif value <= 150:
        return 'bad'
    else:
        return 'worse'

data['pm10_grade'] = data['pm10'].apply(classify_pm10)

# 8-2. 등급 분포 비율
grade_counts = data['pm10_grade'].value_counts().reset_index()
grade_counts.columns = ['pm10_grade', 'count']
grade_counts['percentage'] = (grade_counts['count'] / grade_counts['count'].sum() * 100).round(2)

# 9. 구별 good 등급 비율 상위 5개
good_data = data[data['pm10_grade'] == 'good']
good_ratio = good_data.groupby('district').size().reset_index(name='good_count')
total_ratio = data.groupby('district').size().reset_index(name='total_count')
merged_ratio = pd.merge(good_ratio, total_ratio, on='district')
merged_ratio['good_pct'] = (merged_ratio['good_count'] / merged_ratio['total_count'] * 100).round(2)
top5_good_gu = merged_ratio.sort_values(by='good_pct', ascending=False).head(5)

# [10] 시각화 - 일별 PM10 추이
daily_trend = data.groupby('date')['pm10'].mean().reset_index()

plt.figure(figsize=(14, 6))
plt.plot(daily_trend['date'], daily_trend['pm10'], color='blue')
plt.title('Daily Trend of PM10 in Seoul, 2019')
plt.xlabel('Date')
plt.ylabel('PM10 (㎍/㎥)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("pm10_daily_trend.png")
plt.close()

# [11] 시각화 - 계절별 등급 비율
season_grade_counts = data.groupby(['season', 'pm10_grade']).size().reset_index(name='count')
total_season_counts = data.groupby('season').size().reset_index(name='total')
season_grade_pct = pd.merge(season_grade_counts, total_season_counts, on='season')
season_grade_pct['pct'] = (season_grade_pct['count'] / season_grade_pct['total'] * 100).round(2)

plt.figure(figsize=(10,6))
sns.barplot(data=season_grade_pct, x='season', y='pct', hue='pm10_grade')
plt.title('Seasonal Distribution of PM10 Grades in Seoul, 2019')
plt.ylabel('Percentage (%)')
plt.xlabel('Season')
plt.legend(title='PM10 Grade')
plt.tight_layout()
plt.savefig("pm10_grade_by_season.png")
plt.close()

