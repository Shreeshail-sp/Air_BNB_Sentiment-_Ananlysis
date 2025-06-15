import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import os

# Custom CSS styles
CUSTOM_STYLE = {
    'font-family': 'Segoe UI, Arial, sans-serif',
    'background-color': '#f8f9fa',
    'padding': '20px'
}

NAV_STYLE = {
    'background-color': '#2c3e50',
    'padding': '15px',
    'margin-bottom': '30px',
    'border-radius': '5px',
    'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
}

CARD_STYLE = {
    'background-color': 'white',
    'border-radius': '10px',
    'padding': '20px',
    'margin-bottom': '30px',
    'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
}

def safe_review_length(x):
    try:
        words = len(str(x).split())
        if words < 50:
            return 'Short'
        elif words < 150:
            return 'Medium'
        else:
            return 'Long'
    except Exception:
        return 'Unknown'

# Load the ranked listings data
ranked_df = pd.read_csv('listings_ranked.csv')

# Try to get review_length and sentiment_score from the main data if missing
review_length_available = False
if 'review_length' not in ranked_df.columns:
    try:
        main_df = pd.read_csv('data/reviews_chicago.csv')
        if 'review_length' not in main_df.columns:
            main_df['review_length'] = main_df['comments'].apply(safe_review_length)
        if 'sentiment_score' not in main_df.columns:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            sid = SentimentIntensityAnalyzer()
            main_df['sentiment_score'] = main_df['comments'].apply(lambda x: sid.polarity_scores(str(x))['compound'])
        review_length_df = main_df[['review_length', 'sentiment_score']]
        review_length_available = True
    except Exception as e:
        review_length_available = False
else:
    review_length_available = True
    review_length_df = ranked_df[['review_length', 'sentiment_score']]

# Initialize the Dash app
app = dash.Dash(__name__)

# Create the layout
app.layout = html.Div([
    # Navigation Bar
    html.Div([
        html.H1('Airbnb Sentiment Analysis', 
                style={'color': 'white', 'margin': '0', 'font-size': '24px'}),
        html.Div([
            html.A('Dashboard', href='#', style={'color': 'white', 'margin-right': '20px'}),
            html.A('Word Clouds', href='#wordclouds', style={'color': 'white', 'margin-right': '20px'}),
            html.A('Emoji Analysis', href='#emoji', style={'color': 'white'})
        ], style={'float': 'right'})
    ], style=NAV_STYLE),
    
    # Main Content
    html.Div([
        # Top listings section
        html.Div([
            html.H2('Top Rated Listings', 
                   style={'color': '#2c3e50', 'margin-bottom': '20px', 'font-weight': '600'}),
            dcc.Graph(
                id='top-listings',
                figure=px.bar(
                    ranked_df.head(10),
                    x='listing_id',
                    y='sentiment_score',
                    title='Top 10 Listings by Sentiment Score',
                    labels={'listing_id': 'Listing ID', 'sentiment_score': 'Sentiment Score'},
                    color='sentiment_score',
                    color_continuous_scale='RdYlGn'
                ).update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'family': 'Segoe UI, Arial, sans-serif'},
                    title={'font': {'size': 20, 'color': '#2c3e50'}}
                )
            )
        ], style=CARD_STYLE),
        
        # Sentiment distribution
        html.Div([
            html.H2('Sentiment Distribution', 
                   style={'color': '#2c3e50', 'margin-bottom': '20px', 'font-weight': '600'}),
            dcc.Graph(
                id='sentiment-dist',
                figure=px.histogram(
                    ranked_df,
                    x='sentiment_score',
                    title='Distribution of Sentiment Scores',
                    nbins=50,
                    color_discrete_sequence=['#3498db']
                ).update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'family': 'Segoe UI, Arial, sans-serif'},
                    title={'font': {'size': 20, 'color': '#2c3e50'}}
                )
            )
        ], style=CARD_STYLE),
        
        # Feature Analysis Section
        html.Div([
            html.H2('Feature Analysis', 
                   style={'color': '#2c3e50', 'margin-bottom': '20px', 'font-weight': '600'}),
            html.Div([
                html.Div([
                    html.H3('Positive Features', 
                           style={'color': '#2c3e50', 'font-size': '18px'}),
                    html.A('View Positive Word Cloud', 
                          href='results/positive_wordcloud_interactive.html',
                          target='_blank',
                          style={
                              'display': 'inline-block',
                              'padding': '10px 20px',
                              'background-color': '#2ecc71',
                              'color': 'white',
                              'text-decoration': 'none',
                              'border-radius': '5px',
                              'margin-top': '10px'
                          })
                ], style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'}),
                html.Div([
                    html.H3('Negative Features', 
                           style={'color': '#2c3e50', 'font-size': '18px'}),
                    html.A('View Negative Word Cloud', 
                          href='results/negative_wordcloud_interactive.html',
                          target='_blank',
                          style={
                              'display': 'inline-block',
                              'padding': '10px 20px',
                              'background-color': '#e74c3c',
                              'color': 'white',
                              'text-decoration': 'none',
                              'border-radius': '5px',
                              'margin-top': '10px'
                          })
                ], style={'width': '50%', 'display': 'inline-block', 'text-align': 'center'})
            ])
        ], style=CARD_STYLE),
        
        # Review length analysis
        html.Div([
            html.H2('Review Length Analysis', 
                   style={'color': '#2c3e50', 'margin-bottom': '20px', 'font-weight': '600'}),
            dcc.Graph(
                id='review-length',
                figure=px.box(
                    review_length_df,
                    x='review_length',
                    y='sentiment_score',
                    title='Sentiment Score by Review Length',
                    color='review_length',
                    color_discrete_sequence=px.colors.qualitative.Set3
                ).update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font={'family': 'Segoe UI, Arial, sans-serif'},
                    title={'font': {'size': 20, 'color': '#2c3e50'}}
                ) if review_length_available else go.Figure(),
            ),
            html.Div(
                'Review length data not available.' if not review_length_available else '',
                style={'color': '#e74c3c', 'marginTop': '10px'}
            )
        ], style=CARD_STYLE),
        
        # Emoji analysis
        html.Div([
            html.H2('Emoji Analysis', 
                   style={'color': '#2c3e50', 'margin-bottom': '20px', 'font-weight': '600'}),
            html.A('View Emoji Analysis', 
                  href='results/emoji_analysis.html',
                  target='_blank',
                  style={
                      'display': 'inline-block',
                      'padding': '10px 20px',
                      'background-color': '#3498db',
                      'color': 'white',
                      'text-decoration': 'none',
                      'border-radius': '5px'
                  })
        ], style=CARD_STYLE),
        
        # Search functionality
        html.Div([
            html.H2('Search Listings', 
                   style={'color': '#2c3e50', 'margin-bottom': '20px', 'font-weight': '600'}),
            html.Div([
                dcc.Input(
                    id='search-input',
                    type='text',
                    placeholder='Enter listing ID...',
                    style={
                        'width': '200px',
                        'padding': '10px',
                        'border': '1px solid #ddd',
                        'border-radius': '5px',
                        'margin-right': '10px'
                    }
                ),
                html.Button('Search', 
                           id='search-button', 
                           n_clicks=0,
                           style={
                               'padding': '10px 20px',
                               'background-color': '#2c3e50',
                               'color': 'white',
                               'border': 'none',
                               'border-radius': '5px',
                               'cursor': 'pointer'
                           })
            ], style={'margin-bottom': '20px'}),
            html.Div(id='search-results')
        ], style=CARD_STYLE)
    ], style=CUSTOM_STYLE)
])

# Callback for search functionality
@app.callback(
    Output('search-results', 'children'),
    [Input('search-button', 'n_clicks')],
    [dash.dependencies.State('search-input', 'value')]
)
def update_search_results(n_clicks, search_value):
    if not search_value:
        return html.Div('Enter a listing ID to search', 
                       style={'color': '#7f8c8d', 'font-style': 'italic'})
    
    try:
        listing_id = int(search_value)
        result = ranked_df[ranked_df['listing_id'] == listing_id]
        if len(result) > 0:
            return html.Div([
                html.H3(f'Listing {listing_id} Details', 
                       style={'color': '#2c3e50', 'margin-bottom': '15px'}),
                html.Table([
                    html.Thead(
                        html.Tr([html.Th(col, style={'padding': '10px', 'background-color': '#f8f9fa'}) 
                               for col in result.columns])
                    ),
                    html.Tbody([
                        html.Tr([html.Td(result.iloc[0][col], style={'padding': '10px'}) 
                               for col in result.columns])
                    ])
                ], style={
                    'width': '100%',
                    'border-collapse': 'collapse',
                    'margin-top': '20px',
                    'box-shadow': '0 2px 4px rgba(0,0,0,0.1)'
                })
            ])
        else:
            return html.Div('Listing not found', 
                          style={'color': '#e74c3c', 'font-style': 'italic'})
    except:
        return html.Div('Please enter a valid listing ID', 
                       style={'color': '#e74c3c', 'font-style': 'italic'})

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050) 