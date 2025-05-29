import pandas as pd
url='https://github.com/gahyeok/myproject/blob/master/한국_기업문화_HR_데이터셋_샘플.csv'
korea_df = pd.read_csv(url, sep='\t')  # 탭으로 구분된 CSV 파일 읽기


korea_df1 = korea_df[['Age', '이직여부' ,'일일성과지표', '근무환경만족도', '시간당급여', '이전회사경험수']]

korea_df1 = korea_df1.dropna()

tmp = []
for each in korea_df1['이직여부']:
    tmp.append(0 if each == 'No' else 1)

korea_df1['이직여부1'] = tmp
korea_df1.drop(columns='이직여부', inplace=True)

# 나머지는 다 수치형이여서 인코딩 필요 없음

'''

피처 선택이유
Age: 나이가 많을수록 회사에 안정적으로 있고 싶어서 이직률 적어짐
이직여부: 한번 이직해본 사람이 또 이직할 수 있음. 한번 해보면 그다음은 좀 더 쉽기에
일일성과지표: 성과가 높은 사람은 다른 회사에서도 인재이기에 스카웃될 수 있고 본인도 욕심 낼 가능성 있음
근무환경만족도: 근무 환경에 대해 만족도가 낮으면 이직 확률 높아짐
시간당급여: 급여가 낮을수록 이직 확률 높아짐
이전회사경험수: 겸험수가 적으면 이직확률에 영향을 미치지 않지만 일정 수를 넘으면 또 이직할 가능성이 높아짐

'''

raw = korea_df1
np_raw = raw.values
type(np_raw)


train = np_raw[:800]
test = np_raw[800:]


y_train = [i[0] for i in train]   
X_train = [j[1:] for j in train]  

y_test = [i[0] for i in test]     
X_test = [j[1:] for j in test]    


len(X_train), len(y_train), len(y_test), len(X_test)


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_leaf=5)
model.fit(X_train, y_train)


print('Score:', model.score(X_train, y_train))
print('Score:', model.score(X_test, y_test))


from sklearn.tree import export_graphviz
import graphviz

export_graphviz(
    model,
    out_file="korea.dot",
    feature_names=['Age' ,'일일성과지표', '근무환경만족도', '시간당급여', '이전회사경험수'],
    class_names=['0', '1'],
    rounded=True,
    filled=True
)


with open("korea.dot") as f:
    dot_graph = f.read()

dot = graphviz.Source(dot_graph)
dot.format = 'png'
dot.render(filename='korea_tree', directory='image/decision_trees', cleanup=True)
dot


from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Test Accuracy is ", accuracy_score(y_test, y_pred)*100)



from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


feature_names = ['Age' ,'일일성과지표', '근무환경만족도', '시간당급여', '이전회사경험수']

person1 = []
person2 = []
person3 = []

