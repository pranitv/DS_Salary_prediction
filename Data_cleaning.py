# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 12:35:27 2021

@author: PRANIT
"""
import pandas as pd

df = pd.read_csv('glassdoor_jobs.csv')

#Parsing Salary
df = df[df['Salary Estimate']!='-1']
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided' in x.lower() else 0)

Salary = df['Salary Estimate'].apply(lambda x:x.split('(')[0])

minus_kd = Salary.apply(lambda x:x.replace('K','').replace('$',''))

min_hr = minus_kd.apply(lambda x:x.lower().replace('per hour','').replace('employer provided salary:',''))

df['min_Salary'] = min_hr.apply(lambda x:int(x.split('-')[0]))
df['max_Salary'] = min_hr.apply(lambda x:int(x.split('-')[1]))
df['avg_Salary'] = (df.min_Salary + df.max_Salary)/2

#Company Name Text Only
df['Company_txt'] = df.apply(lambda x:x['Company Name'] if x['Rating']<0 else x['Company Name'][:-3],axis=1)

#State field
df['job_state'] = df['Location'].apply(lambda x:x.split(',')[1])

df['same_state'] = df.apply(lambda x:1 if x.Location==x.Headquarters else 0,axis=1)
#Age of company
df['Age'] = df.Founded.apply(lambda x:x if x<1 else 2021-x)

#Parsing Job descripion
df['python_ya'] = df['Job Description'].apply(lambda x:1 if 'python' in x.lower() else 0)
df['r studio'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
df['spark_ya'] = df['Job Description'].apply(lambda x:1 if 'spark' in x.lower() else 0)
df['aws_ya'] = df['Job Description'].apply(lambda x:1 if 'aws' in x.lower() else 0)
df['excel_ya'] = df['Job Description'].apply(lambda x:1 if 'excel' in x.lower() else 0)

df_out = df.drop(['Unnamed: 0'],axis=1)

df_out.to_csv('Salary_data_cleaned.csv',index=False)