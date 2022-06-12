import pandas as pd
import warnings
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, dash_table


pd.set_option('display.float_format', '{:.2f}'.format)
sd_2019 = pd.read_csv('sharing_data_outside2019.csv')
sd_2020 = pd.read_csv('2020_sharing_data_outside.csv')
sd_2019 = sd_2019.loc[sd_2019['proglang_rank.C++'] == 1]
sd_2020 = sd_2020.loc[sd_2020['proglang_rank.C++'] == 1]

ides2019 = ["cpp_ide.CLion", "cpp_ide.Visual Studio", "cpp_ide.Visual Studio Code", "cpp_ide.NetBeans",
            "cpp_ide.Eclipse CDT", "cpp_ide.QtCreator", "cpp_ide.Xcode", "cpp_ide.Atom", "cpp_ide.Sublime",
            "cpp_ide.Vi/Vim", "cpp_ide.Emacs", "cpp_ide.Other"]

ides2020 = ["ides.CLion", "ides.Visual Studio", "ides.VS Code (Visual Studio Code)", "ides.NetBeans", "ides.Eclipse",
            "ides.QtCreator", "ides.Xcode", "ides.Atom", "ides.Sublime Text", "ides.Vim", "ides.Emacs",
            "ides.Notepad++", "ides.Other"]

sd_2019 = sd_2019[sd_2019[ides2019].notnull().any(1)]
sd_2020 = sd_2020[sd_2020[ides2020].notnull().any(1)]

ide_analysis2019 = pd.DataFrame(columns=['IDE', 'Number of Users'])
for ide in ides2019:
    num_of_users = len(sd_2019.loc[sd_2019[ide].isna() != True])
    ide_analysis2019 = ide_analysis2019.append({"IDE": ide, 'Number of Users': num_of_users}, ignore_index=True)
ide_analysis2019['IDE'] = ide_analysis2019['IDE'].str.replace('cpp_ide.', '', regex=False)

ide_analysis2020 = pd.DataFrame(columns=['IDE', 'Number of Users'])
for ide in ides2020:
    num_of_users = len(sd_2020.loc[sd_2020[ide].isna() != True])
    ide_analysis2020 = ide_analysis2020.append({"IDE": ide, 'Number of Users': num_of_users}, ignore_index=True)
ide_analysis2020['IDE'] = ide_analysis2020['IDE'].str.replace('ides.', '', regex=False)

ide_analysis2019.loc[ide_analysis2019.IDE == 'Visual Studio Code', 'IDE'] = 'VS Code (Visual Studio Code)'

ide_analysis2019.loc[ide_analysis2019.IDE == 'Vi/Vim', 'IDE'] = 'Vim'
ide_analysis2019.loc[ide_analysis2019.IDE == 'Sublime', 'IDE'] = 'Sublime Text'
ide_analysis2019.loc[ide_analysis2019.IDE == 'Eclipse CDT', 'IDE'] = 'Eclipse'

ide_analysis2019['Market Share'] = ide_analysis2019['Number of Users'] / ide_analysis2019['Number of Users'].sum()
ide_analysis2020['Market Share'] = ide_analysis2020['Number of Users'] / ide_analysis2020['Number of Users'].sum()

ide_analysis2019 = ide_analysis2019.sort_values(by=['IDE']).reset_index(drop=True)

ide_analysis_trends = ide_analysis2020.loc[ide_analysis2020.IDE != 'Notepad++'].sort_values(by=['IDE']).reset_index(
    drop=True)

ide_analysis_trends['Market Share Change'] = ide_analysis_trends['Market Share'] - ide_analysis2019['Market Share']

ide_analysis_trends = ide_analysis_trends.sort_values(by='Market Share Change', ascending=False)

# This dataframe has 244 lines, but 4 distinct values for `day`
Market_share_fig_2019 = px.pie(ide_analysis2019, values='Number of Users', names='IDE',
                               title='Распределение долей рынка IDE по результатам Developer Ecosystem Survey 2019')

Market_share_trends_fig_2020 = px.bar(ide_analysis_trends, x='IDE', y='Market Share Change',
                                      title='Изменение долей рынка C++ IDE (DevEcosystem 2019-2020)')

Market_share_trends_fig_2020.update_layout(
    font=dict(
        size=10,
    )
)

# _______________Анализ Выручки

df = pd.read_csv('sales.csv')
df_cl = df.loc[df['product'] == 'CL']

revenue2019 = df_cl.loc[df_cl.date < '2020-01-01']['final_price_usd'].sum()

revenue2020 = df_cl.loc[df_cl.date >= '2020-01-01']['final_price_usd'].sum()

total_revenue_comparison = pd.DataFrame(columns=['Year', 'Revenue'])

total_revenue_comparison = total_revenue_comparison.append({'Year': 2019, 'Revenue': revenue2019}, ignore_index=True)
total_revenue_comparison = total_revenue_comparison.append({'Year': 2020, 'Revenue': revenue2020}, ignore_index=True)

# fig = px.bar(total_revenue_comparison, x='Year', y='Revenue', title='Выручка CLion по годам')
# fig.show()


revenue2020growth = revenue2020 - revenue2019

revenue2019renewal = df_cl[(df_cl.date < '2020-01-01') & (df_cl.license_type == 'Renew')]['final_price_usd'].sum()
revenue2019new = revenue2019 - revenue2019renewal

revenue2020renewal = df_cl[(df_cl.date >= '2020-01-01') & (df_cl.license_type == 'Renew')]['final_price_usd'].sum()
revenue2020new = revenue2020 - revenue2020renewal

revenue2020renewal_growth = revenue2020renewal - revenue2019renewal

revenue2020new_growth = revenue2020new - revenue2019new

revenue_analysis_results = pd.DataFrame(columns=['Year', 'Licence Type', 'Revenue'])

revenue_analysis_results = revenue_analysis_results.append(
    {'Year': "2019", 'Licence Type': "Renewal", 'Revenue': revenue2019renewal}, ignore_index=True)
revenue_analysis_results = revenue_analysis_results.append(
    {'Year': "2019", 'Licence Type': "New", 'Revenue': revenue2019new}, ignore_index=True)
revenue_analysis_results = revenue_analysis_results.append(
    {'Year': "2020", 'Licence Type': "Renewal", 'Revenue': revenue2020renewal}, ignore_index=True)
revenue_analysis_results = revenue_analysis_results.append(
    {'Year': "2020", 'Licence Type': "New", 'Revenue': revenue2020new}, ignore_index=True)

revenue_analysis_results_fig = px.bar(revenue_analysis_results, x="Year", y="Revenue", color="Licence Type",
                                      hover_data=['Licence Type'], barmode='stack',
                                      title='Выручка CLion по годам 2019-2020')

revenue_analysis_results_fig.update_layout(
    font=dict(
        size=10,
    )
)

df_cl_by_date = df_cl[['date', 'final_price_usd']].groupby(by=['date']).sum().reset_index()
df_cl_by_date['date'] = pd.to_datetime(df_cl_by_date['date'])
df_monthly_grouped = df_cl_by_date.groupby(pd.Grouper(freq='M', key='date'))['final_price_usd'].sum().reset_index()

df_monthly_grouped['date'] = df_monthly_grouped['date'].dt.strftime("%B-%Y")

df_monthly_grouped = df_monthly_grouped.rename({'final_price_usd': 'Revenue'}, axis='columns')

monthly_revenue_graph = px.line(df_monthly_grouped, x="date", y="Revenue",
                                title='График ежемесячной выручки CLion с 2019 по конец 2020')
monthly_revenue_graph.update_layout(
    font=dict(
        size=10,
    )
)

# _______________Прогнозирование Выручки
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

from itertools import product
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error

# import packages for hyperparameters tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb

import time
import sys
import gc
import pickle
import random
import datetime

df = pd.read_csv('sales.csv')
df = df.loc[df['product'] == 'CL']
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['date'] = pd.to_datetime(df['date']) - pd.to_timedelta(7, unit='d')
df = df[['date', 'final_price_usd']]
df = df.groupby([pd.Grouper(key='date', freq='W-MON')]).sum().reset_index().sort_values('date')

df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['dayofweek'] = df['date'].dt.dayofweek

df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['dayofyear'] = df['date'].dt.dayofyear
df['dayofmonth'] = df['date'].dt.day
df['weekofyear'] = df['date'].dt.weekofyear

data_grouped = df.drop(columns='date')

split_num = 95
train = data_grouped.iloc[:split_num]
test = data_grouped.iloc[split_num:]

X_train = train.drop(columns='final_price_usd')
Y_train = train['final_price_usd']
X_test = test.drop(columns='final_price_usd')
Y_test = test['final_price_usd']

reg = xgb.XGBRegressor(n_estimators=100000, objective='reg:squarederror')
reg.fit(X_train, Y_train,
        eval_set=[(X_train, Y_train), (X_test, Y_test)],
        early_stopping_rounds=50,
        verbose=False)  # Change verbose to True if you want to see it train

df = pd.read_csv('sales.csv')
df = df.loc[df['product'] == 'CL']
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['date'] = pd.to_datetime(df['date']) - pd.to_timedelta(7, unit='d')
df = df[['date', 'final_price_usd']]

start = datetime.datetime.strptime("01-01-2021", "%d-%m-%Y")
end = datetime.datetime.strptime("31-12-2021", "%d-%m-%Y")
date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end - start).days)]

for date in date_generated:
    df = df.append({'date': date, 'final_price_usd': None}, ignore_index=True)

df = df.groupby([pd.Grouper(key='date', freq='W-MON')]).sum().reset_index().sort_values('date')

df = df.loc[df.date < '2022-01-01']

df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['dayofweek'] = df['date'].dt.dayofweek

df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year
df['dayofyear'] = df['date'].dt.dayofyear
df['dayofmonth'] = df['date'].dt.day
df['weekofyear'] = df['date'].dt.weekofyear

train = df.loc[df.date < '2021-01-01']
test = df.loc[df.date >= '2021-01-01']

X_test = test.drop(columns=['final_price_usd', 'date'])
test['MW_Prediction'] = reg.predict(X_test)
test.final_price_usd = None
result = pd.concat([test, train], sort=False)

prediction_fig = go.Figure()
prediction_fig.add_trace(go.Scatter(x=result.date, y=result.final_price_usd,
                                    mode='lines',
                                    name='Real revenue'))
prediction_fig.add_trace(go.Scatter(x=result.date, y=result.MW_Prediction,
                                    mode='lines',
                                    name='Predicted revenue'))
prediction_fig.update_layout(title_text='Прогноз выручки на 2021 год', title_x=0.5)

bar_prediction_chart = pd.DataFrame(columns=['Year', 'Revenue'])
bar_prediction_chart = bar_prediction_chart.append(
    {'Year': '2019', 'Revenue': train.loc[train.date < '2020-01-01']['final_price_usd'].sum()}, ignore_index=True)
bar_prediction_chart = bar_prediction_chart.append(
    {'Year': '2020', 'Revenue': train.loc[train.date >= '2020-01-01']['final_price_usd'].sum()}, ignore_index=True)
bar_prediction_chart = bar_prediction_chart.append({'Year': '2021', 'Revenue': test['MW_Prediction'].sum()},
                                                   ignore_index=True)

bar_prediction_chart_fig = px.bar(bar_prediction_chart, x='Year', y='Revenue',
                                  title='Прогнозная выручка CLion на 2021 год  в сравнении с реальной')

bar_prediction_chart_fig.update_layout(
    font=dict(
        size=10,
    )
)

# ---------------Keyword Analysis
import nltk
import rake_nltk
from rake_nltk import Rake
import pandas as pd

nltk.download('stopwords')
nltk.download('punkt')

df = pd.read_csv('QueryResults (2).csv')
keywords_dict = dict()
keywords_top_example = dict()
for i in range(0, len(df)):
    rake_nltk_var = Rake()
    text = df.iloc[i]['Title']
    views = df.iloc[i]['ViewCount']
    keywords = ['error', 'failed', 'issue', 'errrors', 'fails', 'fail', 'issues', 'wont', 'cannot',
                'exception', 'exceptions', 'undefined', 'problem', 'problems', 'trouble', 'troubles', 'fault']
    rake_nltk_var.extract_keywords_from_text(text)
    keyword_extracted = rake_nltk_var.get_ranked_phrases()
    key = keyword_extracted[0]
    flag_kw = False
    for kw in keywords:
        if kw in key:
            flag_kw = True
    if flag_kw and len(key.split()) > 1:
        if key not in keywords_dict.keys():
            keywords_dict[key] = views
            keywords_top_example[key] = {'views': views, 'Title': text, 'Body': df.iloc[i]['Body']}
        else:
            keywords_dict[key] += views
            if keywords_top_example[key]['views'] < views:
                keywords_top_example[key] = {'views': views, 'Title': text, 'Body': df.iloc[i]['Body']}
keywords_analysis_df = pd.DataFrame(columns=['Keyword', 'Top question with keyword', 'Views'])
for key, value in keywords_dict.items():
    keywords_analysis_df = keywords_analysis_df.append(
        {'Keyword': key, 'Views': value, 'Top question with keyword': keywords_top_example[key]['Title']}, ignore_index=True)
keywords_analysis_df = keywords_analysis_df.sort_values(by=['Views'], ascending=False)

#---------------Trends Analysis
team_size_analysis2019 = sd_2019[['team_size']].value_counts().reset_index(name='Number of Occurancies')
team_size_analysis2020 = sd_2020[['team_size']].value_counts().reset_index(name='Number of Occurancies')
team_size_analysis2019['Share'] = team_size_analysis2019['Number of Occurancies'] / team_size_analysis2019['Number of Occurancies'].sum()
team_size_analysis2020['Share'] = team_size_analysis2020['Number of Occurancies'] / team_size_analysis2020['Number of Occurancies'].sum()
specs = [[{'type':'domain'}] * 2] * 1

team_size_analysis_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["2019", "2020"],
        specs=specs,
        print_grid=True)


team_size_analysis_fig.add_trace(
        go.Pie(labels=team_size_analysis2019["team_size"],
               values=team_size_analysis2019["Number of Occurancies"],
               showlegend=False,
               textposition='inside',
               textinfo='label+percent'),
         row=1, col=1)
team_size_analysis_fig.add_trace(
        go.Pie(labels=team_size_analysis2020["team_size"],
               values=team_size_analysis2020["Number of Occurancies"],
               showlegend=False,
               textposition='inside',
               textinfo='label+percent'),
         row=1, col=2)
team_size_analysis_fig.update_layout(title="Размер команды разработчиков C++ (DevEcosystem 2019-2020)", title_x=0.5)

sd_2019 = pd.read_csv('sharing_data_outside2019.csv')
sd_2020 = pd.read_csv('2020_sharing_data_outside.csv')
sd_2019 = sd_2019.loc[sd_2019['proglang_rank.C++'] == 1]
sd_2020 = sd_2020.loc[sd_2020['proglang_rank.C++'] == 1]
aport_proglang = ['adopt_proglang.Java','adopt_proglang.C','adopt_proglang.C++','adopt_proglang.Python','adopt_proglang.C#','adopt_proglang.PHP','adopt_proglang.JavaScript','adopt_proglang.Ruby','adopt_proglang.Elixir','adopt_proglang.Crystal','adopt_proglang.Kotlin','adopt_proglang.Swift','adopt_proglang.Objective-C','adopt_proglang.Visual Basic','adopt_proglang.Scala','adopt_proglang.Go','adopt_proglang.HTML / CSS','adopt_proglang.Haskell','adopt_proglang.R','adopt_proglang.SQL(PL/SQL, T-SQL and otherprogramming extensions over SQL)','adopt_proglang.TypeScript','adopt_proglang.Dart','adopt_proglang.CoffeeScript','adopt_proglang.Clojure / ClojureScript','adopt_proglang.Delphi','adopt_proglang.Cobol','adopt_proglang.Groovy','adopt_proglang.Rust','adopt_proglang.Ceylon','adopt_proglang.Perl','adopt_proglang.Assembly','adopt_proglang.Matlab','adopt_proglang.Lua','adopt_proglang.Shell scripting languages(bash/shell/powershell)','adopt_proglang.Julia','adopt_proglang.F#']
adopt_proglang_2019 =sd_2019[aport_proglang].count().reset_index(name='Number of occurancies').sort_values(by='Number of occurancies', ascending = False).iloc[:10]
adopt_proglang_2019['index'] = adopt_proglang_2019['index'].str.replace("adopt_proglang.", '')
aport_proglang.remove('adopt_proglang.SQL(PL/SQL, T-SQL and otherprogramming extensions over SQL)')

aport_proglang.remove('adopt_proglang.Cobol')
aport_proglang.remove('adopt_proglang.Ceylon')
adopt_proglang_2020 = sd_2020[aport_proglang].count().reset_index(name='Number of occurancies').sort_values(by='Number of occurancies', ascending = False).iloc[:10]
adopt_proglang_2020['index'] = adopt_proglang_2020['index'].str.replace("adopt_proglang.", '')
adopt_proglangs = pd.DataFrame(columns=['2019', '2020'])
for i in range(0,5):
    adopt_proglangs = adopt_proglangs.append({'2019': adopt_proglang_2019['index'].iloc[i], '2020': adopt_proglang_2020['index'].iloc[i]}, ignore_index=True)

age_ranges_2019 = ['age_range.17 or younger',
'age_range.18-20',
'age_range.21-29',
'age_range.30-39',
'age_range.40-49',
'age_range.50-59',
'age_range.60 or older']

sd_2019_age_ranges = sd_2019[age_ranges_2019].count().reset_index(name='Number of C++ Programmers').sort_values(by='Number of C++ Programmers', ascending=False)

sd_2019_age_ranges['Age Range Share'] = sd_2019_age_ranges['Number of C++ Programmers'] / sd_2019_age_ranges['Number of C++ Programmers'].sum()
sd_2019_age_ranges['index'] = sd_2019_age_ranges['index'].str.replace("age_range.", '')
sd_2020_age_ranges = sd_2020[['age_range']].value_counts().reset_index(name='Number of C++ Programmers')
sd_2020_age_ranges['Age Range Share'] = sd_2020_age_ranges['Number of C++ Programmers'] / sd_2020_age_ranges['Number of C++ Programmers'].sum()
sd_2019_age_ranges = sd_2019_age_ranges.rename(columns={'index': 'age_range'})
specs = [[{'type':'domain'}] * 2] * 1

age_ranges_fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["2019", "2020"],
        specs=specs,
        print_grid=True)


age_ranges_fig.add_trace(
        go.Pie(labels=sd_2019_age_ranges["age_range"],
               values=sd_2019_age_ranges["Age Range Share"],
               showlegend=False,
               textposition='inside',
               textinfo='label+percent'),
         row=1, col=1)
age_ranges_fig.add_trace(
        go.Pie(labels=sd_2020_age_ranges["age_range"],
               values=sd_2020_age_ranges["Age Range Share"],
               showlegend=False,
               textposition='inside',
               textinfo='label+percent'),
         row=1, col=2)
age_ranges_fig.update_layout(title="Распределение возрастов разработчиков C++ (DevEcosystem 2019-2020)", title_x=0.5)

secondary_languages_2019 = pd.DataFrame(columns=['language', '2nd language occurance', '3rd language occurance'])
secondary_languages_2020 = pd.DataFrame(columns=['language', '2nd language occurance', '3rd language occurance'])
lang_rank2019=['proglang_rank.Java','proglang_rank.C','proglang_rank.Python','proglang_rank.C#','proglang_rank.PHP','proglang_rank.JavaScript','proglang_rank.Ruby','proglang_rank.Kotlin','proglang_rank.Swift','proglang_rank.Objective-C','proglang_rank.Scala','proglang_rank.Go','proglang_rank.SQL(PL/SQL, T-SQL and otherprogramming extensions over SQL)','proglang_rank.Rust','proglang_rank.Haskell','proglang_rank.HTML / CSS','proglang_rank.Elixir','proglang_rank.Crystal','proglang_rank.Visual Basic','proglang_rank.R','proglang_rank.TypeScript','proglang_rank.Dart','proglang_rank.CoffeeScript','proglang_rank.Clojure / ClojureScript','proglang_rank.Delphi','proglang_rank.Cobol','proglang_rank.Groovy','proglang_rank.Perl','proglang_rank.Assembly','proglang_rank.Matlab','proglang_rank.Lua','proglang_rank.Shell scripting languages(bash/shell/powershell)','proglang_rank.Julia','proglang_rank.F#','proglang_rank.Other']
lang_rank2020 = ['proglang_rank.Java','proglang_rank.C','proglang_rank.Python','proglang_rank.C#','proglang_rank.PHP','proglang_rank.JavaScript','proglang_rank.Ruby','proglang_rank.Kotlin','proglang_rank.Swift','proglang_rank.Objective-C','proglang_rank.Scala','proglang_rank.Go','proglang_rank.SQL(PL/SQL, T-SQL and otherprogramming extensions of SQL)','proglang_rank.Rust','proglang_rank.Haskell','proglang_rank.HTML / CSS','proglang_rank.Elixir','proglang_rank.Visual Basic','proglang_rank.R','proglang_rank.TypeScript','proglang_rank.Dart','proglang_rank.Clojure / ClojureScript','proglang_rank.Delphi','proglang_rank.Groovy','proglang_rank.Perl','proglang_rank.Assembly','proglang_rank.Matlab','proglang_rank.Lua','proglang_rank.Shell scripting languages(bash/shell/powershell)','proglang_rank.Julia','proglang_rank.F#','proglang_rank.Other']

for lang in lang_rank2019:
    second = sd_2019[lang].loc[sd_2019[lang] == 2].count()
    third = sd_2019[lang].loc[sd_2019[lang] == 3].count()
    secondary_languages_2019 = secondary_languages_2019.append(
        {'language': lang, '2nd language occurance': second, '3rd language occurance': third}, ignore_index=True)

for lang in lang_rank2020:
    second = sd_2020[lang].loc[sd_2020[lang] == 2].count()
    third = sd_2020[lang].loc[sd_2020[lang] == 3].count()
    secondary_languages_2020 = secondary_languages_2020.append(
        {'language': lang, '2nd language occurance': second, '3rd language occurance': third}, ignore_index=True)

secondary_languages_2019['language'] = secondary_languages_2019['language'].str.replace('proglang_rank.', '')
secondary_languages_2020['language'] = secondary_languages_2020['language'].str.replace('proglang_rank.', '')


secondary_languages_2019["2-nd Language Share"] = secondary_languages_2019['2nd language occurance'] / \
                                                  secondary_languages_2019['2nd language occurance'].sum()


secondary_languages_2020["2-nd Language Share"] = secondary_languages_2020['2nd language occurance'] / \
                                                  secondary_languages_2020['2nd language occurance'].sum()
secondary_languages_2019 = secondary_languages_2019.sort_values(by='2nd language occurance', ascending=False)[
                               ['language', "2-nd Language Share"]].iloc[:5]
secondary_languages_2020 = secondary_languages_2020.sort_values(by='2nd language occurance', ascending=False)[
                               ['language', "2-nd Language Share"]].iloc[:5]

secondary_languages_fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=["2019", "2020"],
    specs=specs,
    print_grid=True)

secondary_languages_fig.add_trace(
    go.Pie(labels=secondary_languages_2019["language"],
           values=secondary_languages_2019["2-nd Language Share"],
           showlegend=False,
           textposition='inside',
           textinfo='label+percent'),
    row=1, col=1)
secondary_languages_fig.add_trace(
    go.Pie(labels=secondary_languages_2020["language"],
           values=secondary_languages_2020["2-nd Language Share"],
           showlegend=False,
           textposition='inside',
           textinfo='label+percent'),
    row=1, col=2)
secondary_languages_fig.update_layout(title="Второй рабочий язык программирования у разработчиков C++", title_x=0.5)




from base64 import b64encode
import io

app = Dash(__name__)

buffer = io.StringIO()
html_bytes = buffer.getvalue().encode()
encoded = b64encode(html_bytes).decode()

app.layout = html.Div(
    children=[
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div(
                                                    html.Img(
                                                        src=app.get_asset_url(
                                                            "logo.svg"
                                                        ),
                                                        className="page-1a",
                                                    )
                                                ),
                                                html.Div(
                                                    [
                                                        html.H6("Jetbrains"),
                                                        html.H5("Анализ и прогнозирование трендов CLion 2020"),
                                                        html.H6("Market Research & Analytics DS Team"),
                                                    ],
                                                    className="page-1b",
                                                ),
                                            ],
                                            className="page-1c",
                                        )
                                    ],
                                    className="page-1d",
                                ),

                            ],
                            className="page-1g",
                        ),
                        html.Div(
                            []),

                        html.Div(
                            [
                                html.Div([html.P('Источники данных: ')],
                                         className="page-1lol"),
                                html.Div(
                                    [
                                        html.H6("Stack Overflow Exchange", className="page-1h"),
                                        html.A("Stack Overflow Public Service",
                                               href="https://data.stackexchange.com/stackoverflow/query/new", ),
                                    ],
                                    className="page-1i",
                                ),
                                html.Div(
                                    [
                                        html.H6("Опрос DevEcosystem 2019", className="page-1h"),
                                        html.A("Диск с данными",
                                               href="https://drive.google.com/drive/folders/1Et4ZzWXvVfW84ylw0fUbQnaJWGZRFiQg", ),
                                    ],
                                    className="page-1i",
                                ),
                                html.Div(
                                    [
                                        html.H6("Опрос DevEcosystem 2020", className="page-1h"),
                                        html.A("Диск с данными",
                                               href="https://drive.google.com/drive/folders/19FRtrWIP3h_aSXmv3IecZ2UeDfu-V6kZ", ),
                                    ],
                                    className="page-1i",
                                ),
                                html.Div(
                                    [
                                        html.H6("Выручки CLion 2019-2020", className="page-1h"),
                                        html.A("csv-file",
                                               href="https://github.com/plavnck/CLion-Sales-Reviews-Analysis", ),
                                    ],
                                    className="page-1i",
                                ),

                            ],
                            className="page-1j",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H6(
                                            "Оценка текущего финансового положения и доли CLion на рынке",
                                            className="page-1h",
                                        ),
                                        html.P(
                                            "Оценка доли CLion на рынке IDE с помощью данных опроса DevEcosystem 2019 и 2020, демонстрация изменения долей рынка IDE для разработчиков C++ в 2020 году по сравнению с данными от 2019 года, анализ изменения суммарной годовой выручки, демонстрация изменения выручек по каждой из лицензий"),
                                    ],
                                    className="page-1k",
                                ),
                                html.Div(
                                    [
                                        html.H6(
                                            "Прогнозирование выручки на 2021 год",
                                            className="page-1h",
                                        ),
                                        html.P(
                                            "Обучение машинной Xgboost-модели для регрессионного анализа и последующего прогноза еженедельной выручки от продаж CLion. Анализ прогноза выручки CLion"),
                                    ],
                                    className="page-1l",
                                ),
                                html.Div(
                                    [
                                        html.H6(
                                            "Анализ трендов среди разработчиков C++ ",
                                            className="page-1h",
                                        ),
                                        html.P(
                                            "Демонстрация основных тенденций, выявленных при сравнении данных DevEcosystem 2019 и DevEcosystem 2020, сравнение среднестатистического возраста разработчика C++, сравнение выбора adopted-language, сравнение изменений во втором и третьем primary языках разработчиков C++"),
                                    ],
                                    className="page-1m",
                                ),
                                html.Div(
                                    [
                                        html.H6(
                                            "Анализ вопросов на Stack Overflow",
                                            className="page-1h",
                                        ),
                                        html.P(
                                            'Keyword-extraction с помощью rake_nltk для каждого заголовка вопроса в Stack Overflow, включающего в себя слова "CLion" и "Clion". Использование ключевых слов и статистики просмотров на Stack Overflow для выявления популярных проблем, обсуждаемых на Stack Overflow пользователями CLion'),
                                    ],
                                    className="page-1l",
                                ),
                            ],
                            className="page-1n",
                        ),
                    ],
                    className="subpage",

                )

            ],
            className="page",

        ),
        # Page 2
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [html.Div(
                                [
                                    html.H6(['1. Анализ доли CLion на рынке'], className="title", ),
                                    html.H6([
                                        'Доли на рынке IDE для C++ разработчиков были рассчитаны с помощью данных DevEcosystem-2020. Использовались исключительно опросные данные от разработчиков, указавших C++ в качестве основного языка для разработки. CLion по результатам опросов стал третьим по популярности IDE на рынке, уступая Visual Studio и VS Code. Доля CLion составила 14,1%'],
                                        className="text")
                                ])
                                , html.Div(
                                [dcc.Graph(
                                    figure={"data": [go.Pie(labels=ide_analysis2019["IDE"],
                                                            values=ide_analysis2019['Number of Users'],
                                                            title='Распределение долей на рынке IDE по результатам DevEcosystem-2020'
                                                            )
                                                     ],
                                            "layout": go.Layout(
                                                height=400,
                                                # title='Распределение долей на рынке IDE по результатам DevEcosystem-2020',
                                                hovermode="closest",
                                                legend={
                                                    "x": 0.16039179104479998,
                                                    "y": 0,
                                                    "bgcolor": "rgb(255, 255, 255, 0)",
                                                    "bordercolor": "rgba(68, 68, 68, 0)",
                                                    "font": {
                                                        "color": "rgb(68, 68, 68)",
                                                        "size": 8,
                                                    },
                                                    "orientation": "h",
                                                    "traceorder": "normal",
                                                },
                                                margin={
                                                    "r": 40,
                                                    "t": 5,
                                                    "b": 30,
                                                    "l": 40,
                                                },
                                                showlegend=True,

                                            ),
                                            }

                                ),
                                ]
                            ),
                                html.Div(
                                    [
                                        html.H6([
                                            "В сравнении с 2019 годом CLion потерял 13% рынка. Подобное могло произойти из-за того, что в 2019 году разработчики в опросе DevEcosystem выбирали один конкретный IDE под конкретный язык программирования. В 2020 году DevEcocsystem допускал выбор нескольких IDE, что могло привести к уменьшению доли CLion по результатам нового опроса относительно результатов прошлого года."],
                                            className="text")

                                    ]
                                ),
                                html.Div([
                                    dcc.Graph(figure=Market_share_trends_fig_2020,
                                              style={'width': '80vh', 'height': '30vh', 'font': 10})
                                ])

                            ],
                            className="subpage",
                        )
                    ]
                )

            ],
            className="page",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [html.Div(
                                [
                                    html.H6(['2. Анализ Revenue CLion за 2020 год'], className="title", ),
                                    html.H6([
                                        'На помесячном графике выручки 2019-2020г., наблюдается тренд роста выручки. Самыми прибыльными месяцами в 2020 году были ноябрь, декабрь и сентябрь, худшая прибыль в году наблюдалась в апреле, августе и июне.'],
                                        className="text")
                                ])
                                , html.Div(
                                [dcc.Graph(
                                    figure=monthly_revenue_graph
                                ),
                                ]
                            ),
                                html.Div(
                                    [html.Div([
                                        html.H6([
                                            "В 2020 году выручка CLion составила 106 123 518$. Относительно 2019г. годовая выручка выросла на 32.79%. Выручка от новых лицензий выросла на 33.38%, от продления лицензии - на 32,2%"],
                                            className="text1")
                                    ], className='six columns'),
                                        html.Div([
                                            dcc.Graph(figure=revenue_analysis_results_fig,
                                                      style={'width': '40vh', 'height': '35vh', 'font': 10})
                                        ], className='six columns')

                                    ], className="thirdPage row"
                                ),

                            ],
                            className="subpage",
                        )
                    ]
                )

            ],
            className="page",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [html.Div(
                                [
                                    html.H6(['3. Прогнозирование выручки в 2021 году'], className="title", ),
                                    html.H6([
                                        'Для прогнозирования выручки была создана регрессионная машинная модель Xgboost, в качестве Loss-функции была использована MSE. Модель была натренирована на данных 2019г и 1,2,3 квартала 2020г. Данные для прогнозирования группировались понедельно для уменьшения шума и увеличения точности модели. Точность прогнозирования суммарной выручки 4 квартала составила 87.2%.'],
                                        className="text")
                                ])
                                , html.Div(
                                [dcc.Graph(
                                    figure=prediction_fig
                                ),
                                ]
                            ),
                                html.Div(
                                    [html.Div([
                                        html.H6([
                                            "По результатам прогноза в 2021 году выручка составит $99.33 млн., таким образом прогнозируется падение годовой выручки на 4.7% относительно значения 2020 года. "],
                                            className="text1")
                                    ], className='six columns'),
                                        html.Div([
                                            dcc.Graph(figure=bar_prediction_chart_fig,
                                                      style={'width': '40vh', 'height': '35vh', 'font': 10})
                                        ], className='six columns')

                                    ], className="thirdPage row"
                                ),

                            ],
                            className="subpage",
                        )
                    ]
                )

            ],
            className="page",
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [html.Div(
                                [
                                    html.H6(['4. Анализ вопросов в Stack Overflow с упоминанием CLion'], className="title", ),
                                    html.H6([
                                        'За 2019 и 2020 год в Stack Overflow было задано 207 вопросов с упоминанием CLion. С помощью nltk_rake был проведен keyword-анализ, который выдал список ключевых фраз в постах на Stack Overflow. Список ключевых фраз был отфильтрован по количеству просмотров вопросов, в которых они фигурировали. Ниже представлена таблица 10 самых популярных ключевых фраз, относящихся к проблемам, которые описывали пользователи CLion в вопросах Stack Overflow. Таблица включает самый популярный вопрос, включающий данную ключевую фразу и число просмотров вопросов включающих данную ключевую фразу на Stack overflow. '],
                                        className="text")
                                ])
                                , html.Div(
                                [html.Div([dash_table.DataTable(
                data=keywords_analysis_df.iloc[:10].to_dict('rows'),
                columns=[{'name': i, 'id': i} for i in keywords_analysis_df.iloc[:10].columns],
                style_header={'backgroundColor': "#D3D3D3",
                              'fontWeight': 'bold',
                              'fontColor': 'white',
                              'textAlign': 'center', 'font-family': 'HelveticaNeue'},
                style_table={'overflowX': 'scroll', 'font-family': 'HelveticaNeue'},
                style_cell={'minWidth': '180px', 'width': '180px',
                        'maxWidth': '180px','whiteSpace': 'normal', 'font-family': 'Raleway'}),
               html.Hr()
        ])
                                ]
                            ),

                            ],
                            className="subpage",
                        )
                    ]
                )

            ],
            className="page",
        ),
html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [html.Div(
                                [
                                    html.H6(['5. Анализ трендов среди разработчиков C++'], className="title", ),
                                    html.H6([
                                        '1) Среднестатистический размер команды разработчиков C++ увеличивается. Сравнивая результаты опросов DevEcosystem-2019 и DevEcosystem-2020, можно заметить, что разработчики стали реже работать в командах из 2-7 человек, но стали чаще работать в более крупных командах. '],
                                        className="text")
                                ])
                                , html.Div([html.Div(
                                            html.Div([
                                                    dcc.Graph(figure=team_size_analysis_fig,
                                                      style={'width': '85vh', 'height': '50vh', 'font': 10})
                                                        ],)

                                            ),

                                    ],

                                ),

                                 html.Div(
                                     [html.Div([
                                         html.H6(["2) Большинство разработчиков хотят изучить языки Rust, Go, Kotlin и C#. Снизилась популярность языка Python среди разработчиков C++, в прошлом году он занимал 5-е место среди adopted-языков по результатам DevEcosystem, но в этом году место Python занял язык Swift. "],className="text")
                                     ], className='six columns',
                                     ),
                                     html.Div([dash_table.DataTable(
                data=adopt_proglangs.to_dict('rows'),
                columns=[{'name': i, 'id': i} for i in adopt_proglangs.columns],
                style_header={'backgroundColor': "#D3D3D3",
                              'fontWeight': 'bold',
                              'fontColor': 'white',
                              'textAlign': 'center', 'font-family': 'HelveticaNeue'},
                style_table={'overflowX': 'scroll', 'font-family': 'HelveticaNeue'},
                style_cell={'minWidth': '100px', 'width': '100px',
                        'maxWidth': '180px','whiteSpace': 'normal', 'font-family': 'Raleway'}),
               html.Hr()], className='six columns')
                                     ], className="thirdPage row")

                            ],
                            className="subpage",
                        )
                    ]
                )

            ],
            className="page",
        ),
html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [html.Div(
                                [
                                    html.H6([
                                        '3) За год по результатам опроса значительно увеличилась доля разработчиков в возрасте 21-29 лет. Кроме того, в 2020 году исчезла в опросе группа разработчиков в возрасте до 18 лет. Небольшой рост наблюдался в долях разработчиков в возрасте 18-20 лет и 50-59 лет, уменьшилась доля разработчиков в возрасте 30-39 лет и 40-49 лет'],
                                        className="text"),
                                    html.Div([
                                                    dcc.Graph(figure=age_ranges_fig,
                                                      style={'width': '85vh', 'height': '40vh', 'font': 10})
                                                        ],)


                                ])
                                ,

                                 html.Div(
                                     [html.Div([
                                         html.H6(["4) Языки Python и C  наиболее популярные вторые рабочие языки среди разработчиков C++. Доли разработчиков, указавших их в качестве второго основного языка, выросли в этом году на 1.2% и 1% для Python и C соответственно. Языки Java, C#, Javascript остались в этом году в топ-5 популярных вторых языков, потеряв в этом году в популярности 0.9%, 0.3% и 1.05% соответственно."],className="text")
                                     ],
                                     ),
                                     html.Div([dcc.Graph(figure=secondary_languages_fig,
                                                      style={'width': '85vh', 'height': '40vh', 'font': 10})
                                                        ],) ],)
                                ]

                        )
                    ],
                            className="subpage"
                )

            ],
            className="page",
        ),



    ]
)  # Page 6

app.run_server(debug=False, port='8052', host='localhost')
