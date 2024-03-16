Project Description This dataset utilizes data from 2014 Major League Baseball seasons in order to develop an algorithm that predicts the number of wins for a given team in the 2015 season based on several different indicators of success. There are 16 different features that will be used as the inputs to the machine learning and the output will be a value that represents the number of wins.

#Import all the necessary libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#Read and study the provided dataset

#Create Dataframe object

df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/Data-Science-ML-Capstone-Projects/master/baseball.csv')
df
W	R	AB	H	2B	3B	HR	BB	SO	SB	RA	ER	ERA	CG	SHO	SV	E
0	95	724	5575	1497	300	42	139	383	973	104	641	601	3.73	2	8	56	88
1	83	696	5467	1349	277	44	156	439	1264	70	700	653	4.07	2	12	45	86
2	81	669	5439	1395	303	29	141	533	1157	86	640	584	3.67	11	10	38	79
3	76	622	5533	1381	260	27	136	404	1231	68	701	643	3.98	7	9	37	101
4	74	689	5605	1515	289	49	151	455	1259	83	803	746	4.64	7	12	35	86
5	93	891	5509	1480	308	17	232	570	1151	88	670	609	3.80	7	10	34	88
6	87	764	5567	1397	272	19	212	554	1227	63	698	652	4.03	3	4	48	93
7	81	713	5485	1370	246	20	217	418	1331	44	693	646	4.05	0	10	43	77
8	80	644	5485	1383	278	32	167	436	1310	87	642	604	3.74	1	12	60	95
9	78	748	5640	1495	294	33	161	478	1148	71	753	694	4.31	3	10	40	97
10	88	751	5511	1419	279	32	172	503	1233	101	733	680	4.24	5	9	45	119
11	86	729	5459	1363	278	26	230	486	1392	121	618	572	3.57	5	13	39	85
12	85	661	5417	1331	243	21	176	435	1150	52	675	630	3.94	2	12	46	93
13	76	656	5544	1379	262	22	198	478	1336	69	726	677	4.16	6	12	45	94
14	68	694	5600	1405	277	46	146	475	1119	78	729	664	4.14	5	15	28	126
15	100	647	5484	1386	288	39	137	506	1267	69	525	478	2.94	1	15	62	96
16	98	697	5631	1462	292	27	140	461	1322	98	596	532	3.21	0	13	54	122
17	97	689	5491	1341	272	30	171	567	1518	95	608	546	3.36	6	21	48	111
18	68	655	5480	1378	274	34	145	412	1299	84	737	682	4.28	1	7	40	116
19	64	640	5571	1382	257	27	167	496	1255	134	754	700	4.33	2	8	35	90
20	90	683	5527	1351	295	17	177	488	1290	51	613	557	3.43	1	14	50	88
21	83	703	5428	1363	265	13	177	539	1344	57	635	577	3.62	4	13	41	90
22	71	613	5463	1420	236	40	120	375	1150	112	678	638	4.02	0	12	35	77
23	67	573	5420	1361	251	18	100	471	1107	69	760	698	4.41	3	10	44	90
24	63	626	5529	1374	272	37	130	387	1274	88	809	749	4.69	1	7	35	117
25	92	667	5385	1346	263	26	187	563	1258	59	595	553	3.44	6	21	47	75
26	84	696	5565	1486	288	39	136	457	1159	93	627	597	3.72	7	18	41	78
27	79	720	5649	1494	289	48	154	490	1312	132	713	659	4.04	1	12	44	86
28	74	650	5457	1324	260	36	148	426	1327	82	731	655	4.09	1	6	41	92
29	68	737	5572	1479	274	49	186	388	1283	97	844	799	5.04	4	4	36	95
#Explore the dataset

df.head()
W	R	AB	H	2B	3B	HR	BB	SO	SB	RA	ER	ERA	CG	SHO	SV	E
0	95	724	5575	1497	300	42	139	383	973	104	641	601	3.73	2	8	56	88
1	83	696	5467	1349	277	44	156	439	1264	70	700	653	4.07	2	12	45	86
2	81	669	5439	1395	303	29	141	533	1157	86	640	584	3.67	11	10	38	79
3	76	622	5533	1381	260	27	136	404	1231	68	701	643	3.98	7	9	37	101
4	74	689	5605	1515	289	49	151	455	1259	83	803	746	4.64	7	12	35	86
df.tail()
W	R	AB	H	2B	3B	HR	BB	SO	SB	RA	ER	ERA	CG	SHO	SV	E
25	92	667	5385	1346	263	26	187	563	1258	59	595	553	3.44	6	21	47	75
26	84	696	5565	1486	288	39	136	457	1159	93	627	597	3.72	7	18	41	78
27	79	720	5649	1494	289	48	154	490	1312	132	713	659	4.04	1	12	44	86
28	74	650	5457	1324	260	36	148	426	1327	82	731	655	4.09	1	6	41	92
29	68	737	5572	1479	274	49	186	388	1283	97	844	799	5.04	4	4	36	95
df.shape
(30, 17)
df.columns
Index(['W', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'RA', 'ER',
       'ERA', 'CG', 'SHO', 'SV', 'E'],
      dtype='object')
print("The dimension of the dataset:",df.shape)
print(f"\nThe column headers in the dataset: {df.columns}")
The dimension of the dataset: (30, 17)

The column headers in the dataset: Index(['W', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO', 'SB', 'RA', 'ER',
       'ERA', 'CG', 'SHO', 'SV', 'E'],
      dtype='object')
It could be observed that the dataset contains 30 rows and 17 columns now out of which one is the variable / feature of interest (W - Nos of wins)

#Check the description of the dataset

df.describe()
W	R	AB	H	2B	3B	HR	BB	SO	SB	RA	ER	ERA	CG	SHO	SV	E
count	30.000000	30.000000	30.000000	30.000000	30.000000	30.000000	30.000000	30.000000	30.00000	30.000000	30.000000	30.000000	30.000000	30.000000	30.000000	30.000000	30.000000
mean	80.966667	688.233333	5516.266667	1403.533333	274.733333	31.300000	163.633333	469.100000	1248.20000	83.500000	688.233333	635.833333	3.956333	3.466667	11.300000	43.066667	94.333333
std	10.453455	58.761754	70.467372	57.140923	18.095405	10.452355	31.823309	57.053725	103.75947	22.815225	72.108005	70.140786	0.454089	2.763473	4.120177	7.869335	13.958889
min	63.000000	573.000000	5385.000000	1324.000000	236.000000	13.000000	100.000000	375.000000	973.00000	44.000000	525.000000	478.000000	2.940000	0.000000	4.000000	28.000000	75.000000
25%	74.000000	651.250000	5464.000000	1363.000000	262.250000	23.000000	140.250000	428.250000	1157.50000	69.000000	636.250000	587.250000	3.682500	1.000000	9.000000	37.250000	86.000000
50%	81.000000	689.000000	5510.000000	1382.500000	275.500000	31.000000	158.500000	473.000000	1261.50000	83.500000	695.500000	644.500000	4.025000	3.000000	12.000000	42.000000	91.000000
75%	87.750000	718.250000	5570.000000	1451.500000	288.750000	39.000000	177.000000	501.250000	1311.50000	96.500000	732.500000	679.250000	4.220000	5.750000	13.000000	46.750000	96.750000
max	100.000000	891.000000	5649.000000	1515.000000	308.000000	49.000000	232.000000	570.000000	1518.00000	134.000000	844.000000	799.000000	5.040000	11.000000	21.000000	62.000000	126.000000
df.dtypes
W        int64
R        int64
AB       int64
H        int64
2B       int64
3B       int64
HR       int64
BB       int64
SO       int64
SB       int64
RA       int64
ER       int64
ERA    float64
CG       int64
SHO      int64
SV       int64
E        int64
dtype: object
There are only 2 types of data in the dataset = float64 and int64

#Check the null (missing) values in the dataset

df.isnull().sum()
W      0
R      0
AB     0
H      0
2B     0
3B     0
HR     0
BB     0
SO     0
SB     0
RA     0
ER     0
ERA    0
CG     0
SHO    0
SV     0
E      0
dtype: int64
It was observed that there are alot of missing values in nearly all the features

#Check for duplicates and drop if found to avoid redundancy 

df[df.duplicated()]
W	R	AB	H	2B	3B	HR	BB	SO	SB	RA	ER	ERA	CG	SHO	SV	E
No duplicate values was found

#Brief about the dataset

df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30 entries, 0 to 29
Data columns (total 17 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   W       30 non-null     int64  
 1   R       30 non-null     int64  
 2   AB      30 non-null     int64  
 3   H       30 non-null     int64  
 4   2B      30 non-null     int64  
 5   3B      30 non-null     int64  
 6   HR      30 non-null     int64  
 7   BB      30 non-null     int64  
 8   SO      30 non-null     int64  
 9   SB      30 non-null     int64  
 10  RA      30 non-null     int64  
 11  ER      30 non-null     int64  
 12  ERA     30 non-null     float64
 13  CG      30 non-null     int64  
 14  SHO     30 non-null     int64  
 15  SV      30 non-null     int64  
 16  E       30 non-null     int64  
dtypes: float64(1), int64(16)
memory usage: 4.1 KB
#Visualization using heatmap

sns.heatmap(df.isnull())
<Axes: >

#Check the number of unique values in the dataset

df.nunique().to_frame("No. of unique values")
No. of unique values
W	24
R	28
AB	29
H	29
2B	22
3B	23
HR	27
BB	29
SO	29
SB	27
RA	30
ER	30
ERA	30
CG	9
SHO	12
SV	20
E	21
#This value should be checked per column as follows:

for i in df.columns:
    print(df[i].value_counts())
    print("\n")
W
68     3
81     2
76     2
74     2
83     2
98     1
84     1
92     1
63     1
67     1
71     1
90     1
64     1
97     1
95     1
100    1
85     1
86     1
88     1
78     1
80     1
87     1
93     1
79     1
Name: count, dtype: int64


R
689    2
696    2
724    1
647    1
650    1
720    1
667    1
626    1
573    1
613    1
703    1
683    1
640    1
655    1
697    1
694    1
656    1
661    1
729    1
751    1
748    1
644    1
713    1
764    1
891    1
622    1
669    1
737    1
Name: count, dtype: int64


AB
5485    2
5575    1
5631    1
5457    1
5649    1
5565    1
5385    1
5529    1
5420    1
5463    1
5428    1
5527    1
5571    1
5480    1
5491    1
5484    1
5467    1
5600    1
5544    1
5417    1
5459    1
5511    1
5640    1
5567    1
5509    1
5605    1
5533    1
5439    1
5572    1
Name: count, dtype: int64


H
1363    2
1497    1
1386    1
1324    1
1494    1
1486    1
1346    1
1374    1
1361    1
1420    1
1351    1
1382    1
1378    1
1341    1
1462    1
1405    1
1349    1
1379    1
1331    1
1419    1
1495    1
1383    1
1370    1
1397    1
1480    1
1515    1
1381    1
1395    1
1479    1
Name: count, dtype: int64


2B
272    3
260    2
289    2
278    2
277    2
274    2
288    2
300    1
292    1
251    1
236    1
265    1
295    1
257    1
243    1
262    1
279    1
294    1
246    1
308    1
303    1
263    1
Name: count, dtype: int64


3B
27    3
39    2
49    2
17    2
32    2
26    2
42    1
48    1
37    1
18    1
40    1
13    1
34    1
30    1
21    1
46    1
22    1
44    1
33    1
20    1
19    1
29    1
36    1
Name: count, dtype: int64


HR
136    2
167    2
177    2
139    1
137    1
148    1
154    1
187    1
130    1
100    1
120    1
145    1
171    1
140    1
198    1
146    1
156    1
176    1
230    1
172    1
161    1
217    1
212    1
232    1
151    1
141    1
186    1
Name: count, dtype: int64


BB
478    2
383    1
461    1
426    1
490    1
457    1
563    1
387    1
471    1
375    1
539    1
488    1
496    1
412    1
567    1
506    1
439    1
475    1
435    1
486    1
503    1
436    1
418    1
554    1
570    1
455    1
404    1
533    1
388    1
Name: count, dtype: int64


SO
1150    2
973     1
1267    1
1327    1
1312    1
1159    1
1258    1
1274    1
1107    1
1344    1
1290    1
1255    1
1299    1
1518    1
1322    1
1119    1
1264    1
1336    1
1392    1
1233    1
1148    1
1310    1
1331    1
1227    1
1151    1
1259    1
1231    1
1157    1
1283    1
Name: count, dtype: int64


SB
69     3
88     2
78     1
82     1
132    1
93     1
59     1
112    1
57     1
51     1
134    1
84     1
95     1
98     1
104    1
70     1
52     1
121    1
101    1
71     1
87     1
44     1
63     1
83     1
68     1
86     1
97     1
Name: count, dtype: int64


RA
641    1
700    1
731    1
713    1
627    1
595    1
809    1
760    1
678    1
635    1
613    1
754    1
737    1
608    1
596    1
525    1
729    1
726    1
675    1
618    1
733    1
753    1
642    1
693    1
698    1
670    1
803    1
701    1
640    1
844    1
Name: count, dtype: int64


ER
601    1
653    1
655    1
659    1
597    1
553    1
749    1
698    1
638    1
577    1
557    1
700    1
682    1
546    1
532    1
478    1
664    1
677    1
630    1
572    1
680    1
694    1
604    1
646    1
652    1
609    1
746    1
643    1
584    1
799    1
Name: count, dtype: int64


ERA
3.73    1
4.07    1
4.09    1
4.04    1
3.72    1
3.44    1
4.69    1
4.41    1
4.02    1
3.62    1
3.43    1
4.33    1
4.28    1
3.36    1
3.21    1
2.94    1
4.14    1
4.16    1
3.94    1
3.57    1
4.24    1
4.31    1
3.74    1
4.05    1
4.03    1
3.80    1
4.64    1
3.98    1
3.67    1
5.04    1
Name: count, dtype: int64


CG
1     7
2     4
7     4
3     3
0     3
5     3
6     3
4     2
11    1
Name: count, dtype: int64


SHO
12    7
10    5
13    3
8     2
9     2
4     2
15    2
21    2
7     2
14    1
18    1
6     1
Name: count, dtype: int64


SV
35    4
41    3
45    3
44    2
48    2
40    2
56    1
28    1
47    1
50    1
54    1
62    1
39    1
46    1
60    1
43    1
34    1
37    1
38    1
36    1
Name: count, dtype: int64


E
88     3
90     3
86     3
93     2
77     2
95     2
122    1
78     1
75     1
117    1
116    1
111    1
94     1
96     1
126    1
85     1
119    1
97     1
101    1
79     1
92     1
Name: count, dtype: int64


#Perform a correlation on the data to see nos of features that are correlated using heatmap

plt.figure(figsize=[25,15],facecolor='green')
sns.heatmap(df.corr(),annot=True)
<Axes: >

#Use visualization to further determine how the data is distributed on the features and display

df.hist(bins=25,figsize=(15,12))

plt.show()

Split the dataset and commence Model training

#Define features (X) and target variable (y)

X = df.drop(['W'], axis=1)
y = df['W']

#Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)     #Random Forest Regressor model

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
Model Evaluation

#Evaluate the model and display the metrics

lr = LinearRegression()
model = lr.fit(X_train, y_train)
print(lr.score(X_test, y_test)) 
0.7876400316149443
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
Mean Squared Error: 51.967916666666675
R-squared: 0.6050569981000633
Predicting Wins

predictions = lr.predict(X_test)
plt.plot(y_test, predictions, 'o')
m, b = np.polyfit(y_test,predictions, 1)
plt.plot(y_test, m*y_test + b)
plt.xlabel("Actual Number of Wins")
plt.ylabel("Predicted Number of Wins")
Text(0, 0.5, 'Predicted Number of Wins')

#Create a user input for Number of Wins prediction and Dataframe as:

user_input = {}  

for column in df.columns:
    if column != 'W':
        user_input[column] = int(input(f"Enter value for {column}: "))

user_input_df = pd.DataFrame([user_input])

wins_predicted = lr.predict(user_input_df)

print("Predicted Number of Wins:", wins_predicted[0])
=======xxxxxxxx===============xxxxxxxxxxxxx=================
