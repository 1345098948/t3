from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import json

app = Flask(__name__)

# Load data from CSV files
cities_df = pd.read_csv('us-cities.csv')
reviews_df = pd.read_csv('amazon-reviews.csv')

# Endpoint to handle the KNN clustering request
@app.route('/data/knn_reviews/stat/knn_reviews', methods=['GET'])
def knn_reviews():
    try:
        # Get request parameters
        classes = int(request.args.get('classes'))
        k = int(request.args.get('k'))
        words = int(request.args.get('words'))

        # Extract relevant data for clustering
        X = cities_df[['lat', 'lng', 'population']]

        # Use KMeans for clustering
        kmeans = KMeans(n_clusters=classes, random_state=42)
        cities_df['cluster'] = kmeans.fit_predict(X)

        # Process clusters and prepare response
        results = []
        for cluster_id in range(classes):
            cluster_mask = (cities_df['cluster'] == cluster_id)
            cluster_cities = cities_df.loc[cluster_mask, 'city'].tolist()
            center_city = get_center_city(cluster_cities)
            popular_words = get_popular_words(cluster_cities, words)
            weighted_avg_score = calculate_weighted_avg_score(reviews_df, cluster_cities)

            result = {
                'class_id': cluster_id,
                'center_city': center_city,
                'cities': cluster_cities,
                'popular_words': popular_words,
                'weighted_avg_score': weighted_avg_score,
            }

            results.append(result)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_center_city(cities):
    # Replace this with your logic to determine the center city
    return cities[0] if cities else None

def get_popular_words(cities, num_words):
    # Replace this with your logic to determine popular words
    return ['i', 'the', 'me']  # Placeholder data


def calculate_weighted_avg_score(reviews_df, cluster_cities):
    # Merge reviews_df with cities_df to get the population information
    merged_df = pd.merge(reviews_df, cities_df[['city', 'population']], on='city', how='inner')

    # Filter for reviews of cities in the current cluster
    cluster_reviews = merged_df[merged_df['city'].isin(cluster_cities)]

    # Calculate weighted average score
    weighted_avg_score = (cluster_reviews['score'] * cluster_reviews['population']).sum() / cluster_reviews[
        'population'].sum()

    return weighted_avg_score

if __name__ == '__main__':
    app.run(debug=True)