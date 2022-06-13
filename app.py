import io
from base64 import b64encode
from helper import *
import warnings
from dash import Dash

warnings.filterwarnings("ignore")
# _______________Market Share Analysis
pd.set_option('display.float_format', '{:.2f}'.format)
sd_2019 = pd.read_csv('sharing_data_outside2019.csv')
sd_2020 = pd.read_csv('2020_sharing_data_outside.csv')
sd_2019 = sd_2019.loc[sd_2019['proglang_rank.C++'] == 1]
sd_2020 = sd_2020.loc[sd_2020['proglang_rank.C++'] == 1]

ide_analysis2019, Market_share_trends_fig_2020 = create_market_share_analysis_fig(sd_2019, sd_2020)

# _______________Revenue Analysis

df = pd.read_csv('sales.csv')
df_cl = df.loc[df['product'] == 'CL']

revenue_analysis_results_fig, monthly_revenue_graph = create_revenue_analysis_figures(df_cl)

# _______________Revenue Prediction


bar_prediction_chart_fig, prediction_fig = predict_revenue(df_cl)

# ---------------Keyword Analysis

query_df = pd.read_csv('QueryResults (2).csv')

keywords_analysis_df = stack_overflow_quesions_keywords_analysis(query_df)

# ---------------Trends Analysis
team_size_analysis_fig = create_team_size_fig(sd_2019, sd_2020)

adopt_proglangs = adopt_languages_analysis(sd_2019, sd_2020)

age_ranges_fig = create_age_ranges_fig(sd_2019, sd_2020)

secondary_languages_fig = create_secondry_languages_fig(sd_2019, sd_2020)

# ---------------Dashboard Creation
app = Dash(__name__)

buffer = io.StringIO()
html_bytes = buffer.getvalue().encode()
encoded = b64encode(html_bytes).decode()

app.layout = create_app_layout(app, ide_analysis2019, Market_share_trends_fig_2020, monthly_revenue_graph,
                               revenue_analysis_results_fig, prediction_fig, bar_prediction_chart_fig,
                               keywords_analysis_df, team_size_analysis_fig, adopt_proglangs,
                               age_ranges_fig, secondary_languages_fig)

app.run_server(debug=False, port='8052', host='localhost')
