import numpy as np
x=np.arange(10)
print(x)
print(x.shape)
print(len(x))
print(x.dtype)

x=np.array([1,4,13,9])
y=np.array([10,9,4,3])
print(x+y)
print(x-y)
print(x*y)
print(x/y)

a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
print(a[1,:])
print(a[:,2])
print(a[0:2,1:3])

b=np.arange(12)
b=b.reshape((3,4))
print(b)
b=b.ravel()
print(b)

b=np.arange(9)
b=b.reshape((3,3))
a=np.arange(5,14)
a=a.reshape((3,3))
print(a)
print(b)
print(np.dot(a,b))

random=np.random.randint(1,100,20)
print(random)
print(random.mean())
print(np.median(random),random.std(),random.min(),random.max())


import pandas as pd
s=pd.Series([1,7,45,62])
print(s.describe())

diagonal=np.diag([1,2,3,4])
print(diagonal)

identity=np.eye(4,dtype=int)
print(identity)

#pandas
students={
    "Name":["Aniket","Tamojit","Sarasij","Rony","Snehasis"],
    "Age":[21,21,20,22,26],
    "Marks":[100,89,26,34,79]
}

df=pd.DataFrame(students)
print(df)
print(df.head())
print(df.tail())

print(df.iloc[0,1])
print(df.loc[2,['Name','Age']])


def assignGrade(mark):
    if mark>70:
        return 'A'
    else:
        return 'B'
df['Grade']=df['Marks'].apply(assignGrade)
print(df)

df.drop(columns=['Age'],inplace=True)
print(df)

original=pd.DataFrame(students)
# original.to_csv('Originaldata.csv',index=False)

readcsv=pd.read_csv('Originaldata.csv')
print(readcsv)

filteredData=readcsv[readcsv['Marks']>70]
print(filteredData)
# filteredData.to_csv('FilteredData.csv',index=False)

conditionedData=df[df['Marks']>75]
print(conditionedData[['Name','Grade']])

employes={
    "employee":["Aniket","tamojit","sarasij","rony"],
    "department":["Ds","Ds","ECE","It"],
    "salary":[7848,4145,4223,5856]
}
df1=pd.DataFrame(employes)
print(df1)
departmentGrouped=df1.groupby('department')
print(departmentGrouped['salary'].agg(['mean']))

mis={
    "col1":[7,15,63,np.nan,None],
    "col2":[np.nan,78,15,None,96],
    "col3":[7,96,np.nan,19,89]
}
df2=pd.DataFrame(mis)
print(df2)
df2.dropna(axis='index',how='any',inplace=True)
print(df2)
df3=pd.DataFrame(mis)
df3.fillna(0,inplace=True)
print(df3)

#matplotlib
import matplotlib.pyplot as plt
x=[1,2,3,4]
y=[2,4,6,8]
plt.plot(x,y)
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.title("line graph")
plt.show()

product=['Maggi','Surf excel','rice','tea']
sales=[7896,341,88,693]
plt.bar(product,sales)
plt.xlabel("products")
plt.ylabel("sales")
plt.title("sales vs products")
plt.show()

histRand=np.random.randint(0,101,50)
plt.hist(histRand)
plt.show()

a=np.random.randint(1,100,20)
b=np.random.randint(1,100,20)
plt.scatter(a,b)
plt.show()

activities=['sleep','leisure','exercise','study']
time_hours = [8, 6, 6, 4] # Total: 24 hours
plt.pie(time_hours,
        labels=activities, # Use the new labels list here
        autopct='%1.0f%%',       # Display the percentage inside the slice
        startangle=90,
)
plt.show()