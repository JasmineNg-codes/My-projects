#----------------------------------------------------------------------------------------------------------------------------------------------

%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#First, I would like the set a readable size for my graph, as well as the x-axis from 70Ma to present from left to right.
#It is conventional for geoscientists to visualize time moving forwards in time towards the right. 

plt.figure(figsize=(20,10))
plt.xlim(70,0)

#Next,I imported the glob module from the glob method.
#This return a list of file in the current directory for files ending with .txt. I named this list 'files'.
    #reference: https://stackoverflow.com/questions/22431921/abbreviate-the-import-of-multiple-files-with-loadtxt-python
#Then, I created an empty list named 'my list'. I looped the 'files' list for np.loadtxt.
#I appended all the loaded files into the empty list

from glob import glob

files=glob("*.txt")
mylist=[]
for i in files:
    mylist.append(np.loadtxt(i,skiprows=1))

#'my list' now contains many arrays. I concatenated the arrays into one array and named it allsites
#Because the 'allsites' array is not ordered with increasing age, if I wanted to use the data to make 
#a scatterplot with the markers connected, it will give me wrongly connected points. 
#I will sort the data according to the ages in column 0, such that the O18 data
#will remain assigned to the age point in the original files. I named this array 'a'.
    #reference :https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column

allsites=np.concatenate(mylist)
a=allsites[allsites[:, 0].argsort()] 

#I will now create two loops that will read and extract age from column 0 and O18 from column 1 in array a.
#I named the two lists age and O18, then plotted them together with the appropriate formatting and labels.
    #reference: https://stackoverflow.com/questions/903853/how-do-you-extract-a-column-from-a-multi-dimensional-array

age=[row[0] for row in a]
O18=[row[1] for row in a]
plt.plot(age,O18,"-",color="black")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Age(Ma)",fontsize=14)
plt.ylabel("\u03B418O(‰)",fontsize=14)

plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------
#Before entering code for the second question, first I will insert code from question 1 relevant to this section
plt.figure(figsize=(20,10))
plt.ylim(-1,6)
plt.xlim(70,0)
plt.plot(age,O18,"-",color="black")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Age(Ma)",fontsize=14)
plt.ylabel("\u03B418O(‰)",fontsize=14)

#Given that the relationship between 𝛿18O and temperature is related by the equation 𝑇=12−4𝛿18 O,
#I will calculate the temperature limits by inputting the upper and lower limits of the 𝛿18O y-axis into the equation.

T_uplim=12-4*(6)
print("The upper limit for the temperature axis is ", T_uplim)

T_lowlim=12-4*(-1)
print("The lower limit for the temperature axis is ", T_lowlim)

#Now that I know the upper limit and lower limit of the temperature axis is -12 and 16, I will show tick marks
#for only the values in the given list of 0-12.

plt.twinx()
plt.ylim(16,-12)
plt.yticks([12,11,10,9,8,7,6,5,4,3,2,1,0],fontsize=14)
plt.ylabel("Temperature(°C)",fontsize=14)

plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------
#First, I will insert code from question 1 and 2 relevant to this section
plt.figure(figsize=(20,10))
plt.ylim(-1,6)
plt.xlim(70,0)
plt.plot(age,O18,"-",color="black")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Age(Ma)",fontsize=14)
plt.ylabel("\u03B418O(‰)",fontsize=14)

plt.twinx()
plt.ylim(16,-12)
plt.yticks([12,11,10,9,8,7,6,5,4,3,2,1,0],fontsize=14)
plt.ylabel("Temperature(°C)",fontsize=14)

#Here, I plotted lines and labels corresponding to the given ages
plt.axvline(x=65,color="black")
plt.text(63, -12.3, "Paleocene",fontsize=15)
plt.axvline(x=56,color="black")
plt.text(50, -12.3, "Eocene",fontsize=15)
plt.axvline(x=33.9,color="black")
plt.text(31, -12.3, "Oligocene",fontsize=15)
plt.axvline(x=23,color="black")
plt.text(18, -12.3, "Miocene",fontsize=15)
plt.axvline(x=5.33,color="black")
plt.text(6, -12.3, "Pliocene",fontsize=15)
plt.axvline(x=2.58,color="black")

plt.show()
#----------------------------------------------------------------------------------------------------------------------------------------------
#To create a table storing my O18 values for each epoch, I will first install a useful package.
    #reference: https://towardsdatascience.com/how-to-easily-create-tables-in-python-2eaea447d8fd

!pip install tabulate
from tabulate import tabulate

#Next, I will define a function to calculate the mean. The mean values are rounded to four decimal points.

def mean(x):
    return round(sum(x)/len(x),4)

#I will not be able to use np.logical_and() if'age' and 'O18' are lists
#Now, I will convert my 'age', 'O18' lists into arrays

age_arr=np.array(age)
O18_arr=np.array(O18)

#Using np.logical_and(), I will assign an epoch name to the corresponding section of the array.
#The mean function is used to calculate the mean O18 values for the specified age range

pal = np.logical_and(age_arr>56,age_arr<65)   
meanpal=mean(O18_arr[pal])

eo = np.logical_and(age_arr>33.9,age_arr<56)   
meaneo=mean(O18_arr[eo])

oli = np.logical_and(age_arr>23,age_arr<33.9)   
meanoli=mean(O18_arr[oli])

mio = np.logical_and(age_arr>5.33,age_arr<23)   
meanmio=mean(O18_arr[mio])

plio = np.logical_and(age_arr>2.58,age_arr<5.33)   
meanplio=mean(O18_arr[plio])

#To create a table, I have made two lists each containing the mean values and the epoch names. 

meanvalues=["Mean dO18(‰)",meanpal,meaneo,meanoli,meanmio,meanplio]
meanwords=["Epoch","Paleocene","Eocene","Oligocene","Miocene","Pliocene"]

#I will zip them together into two columns,and make it into a table using tabulate
    #reference: https://stackoverflow.com/questions/41468116/python-how-to-combine-two-flat-lists-into-a-2d-array

meantable=np.array(list(zip(meanwords, meanvalues))) 
print(tabulate(meantable,headers='firstrow',tablefmt='grid'))

#----------------------------------------------------------------------------------------------------------------------------------------------

#First, I will insert code from question 1 relevant to this section. I will include previous code from question 2 
#and 3 for the secondary axis and associated lines after plotting the trend for 𝛿18 O.

plt.figure(figsize=(20,10))
plt.ylim(-1,6)
plt.xlim(70,0)
plt.plot(age,O18,"-",color="black")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Age(Ma)",fontsize=14)
plt.ylabel("\u03B418O(‰)",fontsize=14)

#Here, I will use stats.linregress() to calculate a linear model for my age and 𝛿18 O data. 
#I rounded my data to four decimals ponts.

m,c,_,_,_=stats.linregress(age,O18)
m1=round(m,4)

#As python is not aware that time 0 is present, the O18 trend(m2) will give me a reversed sign, which will 
#inaccurately tell me how O18 is changing with increasing time.
#In order to solve this issue, I will define a function to change all positive signs to negative signs and vice versa.

def convert(x):
    return x*-1

m2=convert(m1)

#I would like all the m1 values to display its +/- sign, so I can show whether it is a positive or negative trend
#The following line of code does precisely this. The f string allows for the formatting within {} brackets. 
#In this case, the plus sign means to show the sign of the m2 number regardless if it is positive or negative. 
    #reference: https://bobbyhadz.com/blog/python-print-sign-of-number

m3=f'{m2:+}' 

c1=round(c,4)

#The following lines will specify the x values of the line to be the length of the age data
agefit=np.arange(len(age))
O18fit = c1 + m1*agefit

#Then I will plot my fitted line. The label will specify the trend(m3) calculated.
plt.plot(agefit,O18fit,linewidth=5,label='Trend: '+m3+' \u03B4O18/Ma')

#To ensure the epoch lines that I will plot won't overlap my legend, I will use the bbox_to_anchor keyword
#to place the legend outside of my graph with the specified coordinates
    #reference: https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html
plt.legend(bbox_to_anchor=(1.0, 1.0),loc='upper left',fontsize=16)

#Here I will rename my m3 variable 'trend' for the next question
trend=m3

#Finally, I will insert code from questions 2,3 relevant to this section
plt.twinx()
plt.ylim(16,-12)
plt.yticks([12,11,10,9,8,7,6,5,4,3,2,1,0],fontsize=14)
plt.ylabel("Temperature(°C)",fontsize=14)

plt.axvline(x=65,color="black")
plt.text(63, -12.3, "Paleocene",fontsize=15)
plt.axvline(x=56,color="black")
plt.text(50, -12.3, "Eocene",fontsize=15)
plt.axvline(x=33.9,color="black")
plt.text(31, -12.3, "Oligocene",fontsize=15)
plt.axvline(x=23,color="black")
plt.text(18, -12.3, "Miocene",fontsize=15)
plt.axvline(x=5.33,color="black")
plt.text(6, -12.3, "Pliocene",fontsize=15)
plt.axvline(x=2.58,color="black")

plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------

#First, I will insert code from the preivous questions relevant to this section. 
plt.figure(figsize=(20,10))
plt.ylim(-1,6)
plt.xlim(70,0)
plt.plot(age,O18,"-",color="black")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Age(Ma)",fontsize=14)
plt.ylabel("\u03B418O(‰)",fontsize=14)

#Here I have defined a function with variables u,q,l
#inputing u,q,l - age of the epoch, O18 of the epoch, and epoch name- will output a linear regression plot

def trendfunc(u,q,l):
    m,c,_,_,_=stats.linregress(u,q)
    m1=round(m,4)
    m2=convert(m1)
    m3=f'{m2:+}' 
    c1=round(c,4)
    
    #The following code specifies that the linear regression will be plotted between 
    #minimum and maximum values of u, or age. This will give me a linear regression specific to the epoch.
    
    u1=np.arange(min(u),max(u),1)
    q=c1+m1*u1
    
    #Then the linear regresssion plot is plotted. The label will specify the epoch name(l) and trend(m3).
    plt.plot(u1,q,linewidth=5,label=l+' Trend: '+m3+' \u03B4O18/Ma')
    return(m3)

#Next, I have defined the epoch limits, and used the function to create 5 linear regressions for the epochs

pal = np.logical_and(age_arr>56,age_arr<65) 
trend1 =trendfunc(age_arr[pal],O18_arr[pal],"Paleocene")

eo = np.logical_and(age_arr>33.9,age_arr<56)   
trend2 =trendfunc(age_arr[eo],O18_arr[eo],"Eocene")

oli = np.logical_and(age_arr>23,age_arr<33.9)   
trend3 =trendfunc(age_arr[oli],O18_arr[oli],"Oligocene")

mio = np.logical_and(age_arr>5.33,age_arr<23)   
trend4 =trendfunc(age_arr[mio],O18_arr[mio],"Miocene")

plio = np.logical_and(age_arr>2.58,age_arr<5.33)   
trend5 =trendfunc(age_arr[plio],O18_arr[plio],"Pliocene")

plt.legend(bbox_to_anchor=(1.0, 1.0),loc='upper left',fontsize=16)

#I will insert code from questions 2,3 relevant to this section
plt.twinx()
plt.ylim(16,-12)
plt.yticks([12,11,10,9,8,7,6,5,4,3,2,1,0],fontsize=14)
plt.ylabel("Temperature(°C)",fontsize=14)

plt.axvline(x=65,color="black")
plt.text(63, -12.3, "Paleocene",fontsize=15)
plt.axvline(x=56,color="black")
plt.text(50, -12.3, "Eocene",fontsize=15)
plt.axvline(x=33.9,color="black")
plt.text(31, -12.3, "Oligocene",fontsize=15)
plt.axvline(x=23,color="black")
plt.text(18, -12.3, "Miocene",fontsize=15)
plt.axvline(x=5.33,color="black")
plt.text(6, -12.3, "Pliocene",fontsize=15)
plt.axvline(x=2.58,color="black")

plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------------

#Using the same tabulate method, I have noted down the trends shown on the graph.

O18values=["dO18(‰) per Ma",trend,trend1,trend2,trend3,trend4,trend5]
O18words=["Epoch","Whole period","Paleocene","Eocene","Oligocene","Miocene","Pliocene"]
trendtable=np.array(list(zip(O18words, O18values))) 
print(tabulate(trendtable,headers='firstrow',tablefmt='grid'))

#----------------------------------------------------------------------------------------------------------------------------------------------
