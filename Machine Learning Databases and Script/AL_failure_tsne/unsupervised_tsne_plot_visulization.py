import matplotlib
import numpy as np
import pandas as pd
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, pairwise_distances
import itertools
import optuna
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler  # New import for feature scaling
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

matplotlib.use('Agg')  # Use a non-interactive backend
# Define element information (abbreviated here; include all elements as needed)
element_information={}
element_information['None']=[0,0,0,0,0,0,0,0]
element_information['Ru']=[101.07, 44, 5, 8, 7.5, 2.2, 7, 134]
element_information['Ir']=[192.217, 77, 6, 9, 9.1, 2.2, 7, 136]
element_information['Mn']=[54.938, 25, 4, 7, 7.4, 1.56, 5, 127]
element_information['Ba']=[137.327, 56, 6, 2, 5.19, 0.89, 10, 222]
element_information['Sr']=[87.62, 38, 5, 2, 5.67, 0.95, 10, 215]
element_information['Na']=[22.9897, 11, 3, 1, 5.12, 0.93, 0, 190]
element_information['Ag']=[107.868, 47, 5, 11, 7.54, 1.93, 10, 144]
element_information['La']=[138.905, 57, 6, 3, 5.5, 1.1, 1, 187]
element_information['Zn']=[65.38, 30, 4, 12, 9.35, 1.65, 10, 138]
element_information['K']=[39.0983, 19, 4, 1, 4.32, 0.82, 0, 235] 
element_information['Al']=[26.9815, 13, 3, 13, 5.95, 1.61, 0,143]
element_information['Au']=[196.966, 79, 6, 11, 9.19, 2.54, 10, 144]
element_information['Pr']=[140.904, 59, 6, 3, 5.76, 1.13, 10, 182]
element_information['Nb']=[92.906, 41, 5, 5, 6.76, 1.6, 4, 146]
element_information['Li']=[6.941, 3, 2, 1, 5.37, 0.98, 0, 145]
element_information['Ca']=[40.078, 20, 4, 2, 6.09, 1, 0, 197]
element_information['Cr']=[51.996, 24, 4, 6, 6.74, 1.66, 4, 130]
element_information['In']=[114.818, 49, 5, 13, 8.95, 1.78, 10, 166]
element_information['Nd']=[144.242, 60, 6, 3, 6.31, 1.14, 10, 182]
element_information['Mo']=[95.94, 42, 5, 6, 7.35, 2.16, 5, 139]
element_information['Ti']=[47.867, 22, 4, 4, 6.81, 1.54, 2, 147]
element_information['W']=[183.84, 74, 6, 6, 7.98, 2.36, 4, 141]
element_information['Zr']=[91.224, 40, 5, 4, 6, 1.33, 2, 160]
element_information['Ce']=[140.116, 58, 6, 3, 6.91, 1.12, 1, 181]
element_information['Re']=[186.207, 75, 6, 7, 7.88, 1.9, 5, 137]
element_information['Ta']=[180.947,73, 6, 5, 7.89, 1.5, 3, 149]
element_information['Gd']=[157.25, 64, 6, 3, 6.65, 1.2, 1, 179]
element_information['F']=[18.9984, 9, 2, 17, 18.6, 3.98, 0, 73]
element_information['Sm']=[150.36, 62, 6, 3, 6.55, 1.1, 10, 181]
element_information['N']=[14.0067, 7, 2, 15, 14.48, 3.04, 0, 92]
element_information['Er']=[167.529, 68, 6, 3, 6.108, 1.23, 10, 178]
element_information['Sn']=[118.71, 50, 5, 14, 7.37, 1.96, 10, 162]
element_information['Pd']=[106.42, 46, 5, 10, 8.3, 2.2, 10, 137]
element_information['Ni']=[58.6934, 28, 4, 10, 7.61, 1.91, 8, 124]
#################################################################
element_information['Sc']=[44.956, 21, 4, 3, 6.57, 1.36, 1, 162]
element_information['V']=[50.942, 23, 4, 5, 6.76, 1.63, 3, 134]
element_information['Fe']=[55.845, 26, 4, 8, 7.83, 1.83, 6, 126]
element_information['Co']=[58.933, 27, 4, 9, 7.81, 1.88, 7, 125]
element_information['Cu']=[63.546, 29, 4, 11, 7.69, 1.9, 10, 128]
element_information['Ga']=[69.723, 31, 4, 13, 5.97, 1.81, 10, 141]
element_information['Y']=[88.905, 39, 5, 3, 6.5, 1.22, 1, 178]
element_information['Mo']=[95.94, 42, 5, 6, 7.35, 2.16, 5, 139]
element_information['Tc']=[98.906, 43, 5, 7, 7.28, 1.9, 5, 136]
element_information['Rh']=[102.905, 45, 5, 9, 7.7, 2.28, 8, 134]
element_information['Cd']=[112.411, 48, 5, 12, 8.95, 1.69, 10, 154]
element_information['Pm']=[144.912, 61, 6, 3 ,5.55, 1.13, 10, 183]
element_information['Eu']=[151.964, 63, 6, 3, 5.67, 1.2, 10, 199]
element_information['Tb']=[158.925, 65, 6, 3, 6.74, 1.2, 10, 180]
element_information['Dy']=[162.5, 66, 6, 3, 6.82, 1.22, 10, 180]
element_information['Ho']=[164.93, 67, 6, 3, 6.022, 1.23, 10, 179]
element_information['Tm']=[168.934, 69, 6, 3, 6.184, 1.25, 10, 177]
element_information['Yb']=[173.04, 70, 6, 3, 7.06, 1.1, 10, 176]
element_information['Lu']=[174.967, 71, 6, 3, 5.4259, 1.27, 1, 175]
element_information['Hf']=[178.49, 72, 6, 4, 6.8251, 1.3, 2, 167]
element_information['Os']=[190.23, 76, 6, 8, 8.7, 2.2, 6, 135]
element_information['Pt']=[195.084, 78, 6, 10, 8.9, 2.28, 9, 139]
element_information['Hg']=[200.59, 80, 6, 12, 10.39, 2, 10, 157]
element_information['Tl']=[204.383, 81, 6, 13, 6.08, 1.62, 10, 171]
element_information['Pb']=[207.2, 82, 6, 14, 7.38, 2.33, 10, 175]
element_information['Bi']=[208.98, 83, 6, 15, 7.25, 2.02, 10, 170]
element_information['Mg']=[24.3050, 12, 3, 2, 7.61, 1.31, 0, 160]
element_information['C']=[12.0107, 6, 2, 14, 11.22, 2.56, 0, 77]
element_information['B']=[10.811, 5, 2, 13, 8.33, 2.04, 0, 98]
element_information['P']=[30.9737, 15, 3, 15, 10.3, 2.19, 0, 128]
element_information['S']=[32.065, 16, 3, 16, 10.31, 2.58, 0, 127]
element_information['Sb']=[121.760, 51, 5, 15, 8.35, 2.05, 10, 159]
element_information['Te']=[127.6, 52, 5, 16, 9.0096, 2.1, 10, 160]
element_information['Br']=[79.904, 35, 4, 17, 11.8, 2.96, 10, 115]
element_information['Cl']=[35.453, 17, 3, 17, 12.96, 3.16, 0, 99]
element_information['Si']=[28.0855, 14, 3, 14, 8.12, 1.9, 0, 132]
element_information['Se']=[78.96, 34, 4, 16, 9.5, 2.55, 10, 140]
element_list=list(element_information.keys())
def get_encoded_input(ele_1, ele_2, ele_3, ele_4, prop_1, prop_2, prop_3, prop_4, hydro_temp, hydro_time, hydro_reduc, hydro_ball, post_process):
    """
    Encodes the input values into the format used by the models for predictions.
    
    Parameters:
    - ele_1 to ele_4: Element strings (e.g., 'Ru', 'Ir')
    - prop_1 to prop_4: Proportion values for each element
    - hydro_temp, hydro_time, hydro_reduc, hydro_ball, post_process: Process parameters
    
    Returns:
    - Encoded input as a pandas DataFrame
    """
    test_info = [0.5, 0, 0, 0, 5, 1]
    
    # Retrieve element properties
    ele_info = (element_information.get(ele_1, element_information['None']) + 
                element_information.get(ele_2, element_information['None']) + 
                element_information.get(ele_3, element_information['None']) + 
                element_information.get(ele_4, element_information['None']))
    
    # Add proportions and process conditions
    prop_info = [prop_1, prop_2, prop_3, prop_4]
    hydrothermal_info = [hydro_temp, hydro_time, 1, 0, hydro_reduc, hydro_ball]
    annealing_info = [500, 120, 0, 0, 0, post_process]
    
    # Combine all information into a single input vector
    info_all = ele_info + prop_info + hydrothermal_info + annealing_info + test_info
    input_array = np.array(info_all).reshape(1, -1)
    
    # Convert to DataFrame with column names matching original data format
    input_df = pd.DataFrame(input_array, columns=[f'Feature_{i}' for i in range(len(info_all))])

    return input_df

def process_csv_files():
    print('Processing CSV files...')
    """
    Reads CSV files named from 1st to 5th in the current folder, 
    encodes the input values, and saves the encoded DataFrame to a new CSV file.
    
    Returns:
    - List of tuples containing encoded input DataFrame and sample name
    """
    results = []
    for i in range(1, 6):
        file_name = f'{i}st_final.csv' if i == 1 else f'{i}nd_final.csv' if i == 2 else f'{i}rd_final.csv' if i == 3 else f'{i}th_final.csv'
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            encoded_rows = []
            for index, row in df.iterrows():
                encoded_input = get_encoded_input(
                    row['Metal_Dopant_1'], row['Metal_Dopant_2'], row['Metal_Dopant_3'], row['Metal_Dopant_4'],
                    row['Metal_Dopant_1_Proportion_Precursor'], row['Metal_Dopant_2_Proportion_Precursor'], 
                    row['Metal_Dopant_3_Proportion_Precursor'], row['Metal_Dopant_4_Proportion_Precursor'],
                    row['Hydrothermal/Mixing Temperature (degree)'], row['Hydrothermal/Mixing Time (min)'], 
                    row['Hydrothermal/Mixing Strong Reductant in Liquid'], row['Hydrothermal/Mixing Weak Reductant in Liquid'], 
                    row['Post ProcessingAcid Wash/Leaching/Quenching/Ultrasonication']
                )
                sample_name = row['Sample Name']
                category_label = row['Category_Label']
                overpotential = row['Overpotential mV @10 mA cm-2']
                encoded_input['Sample Name'] = sample_name
                encoded_input['Category_Label'] = category_label
                encoded_input['Overpotential mV @10 mA cm-2'] = overpotential
                encoded_rows.append(encoded_input)
                results.append((encoded_input, sample_name))
            
            # Save the encoded DataFrame to a new CSV file
            encoded_df = pd.concat(encoded_rows, ignore_index=True)
            encoded_df.to_csv(f'{i}st_final_encoded.csv' if i == 1 else f'{i}nd_final_encoded.csv' if i == 2 else f'{i}rd_final_encoded.csv' if i == 3 else f'{i}th_final_encoded.csv', index=False)
    
    return results

def load_encoded_csv():
    print('Loading encoded CSV files...')
    """
    Loads encoded CSV files named from 1st to 5th in the current folder.
    
    Returns:
    - List of tuples containing encoded input DataFrame and sample name
    """
    results = []
    for i in range(1, 6):
        file_name = f'{i}st_final_encoded.csv' if i == 1 else f'{i}nd_final_encoded.csv' if i == 2 else f'{i}rd_final_encoded.csv' if i == 3 else f'{i}th_final_encoded.csv'
        if os.path.exists(file_name):
            df = pd.read_csv(file_name)
            for index, row in df.iterrows():
                encoded_input = row.drop(['Sample Name', 'Category_Label', 'Overpotential mV @10 mA cm-2']).values.reshape(1, -1)
                encoded_df = pd.DataFrame(encoded_input, columns=[f'Feature_{i}' for i in range(encoded_input.shape[1])])
                encoded_df['Sample Name'] = row['Sample Name']
                encoded_df['Category_Label'] = row['Category_Label']
                encoded_df['Overpotential mV @10 mA cm-2'] = row['Overpotential mV @10 mA cm-2']
                encoded_df['Origin'] = f'{i}st' if i == 1 else f'{i}nd' if i == 2 else f'{i}rd' if i == 3 else f'{i}th'
                results.append((encoded_df, row['Sample Name']))
    
    return results

def calculate_separation_score(tsne_results, binary_labels, overpotentials):
    """Calculate a score that measures both clustering quality and separation of failed samples"""
    # Calculate silhouette score
    silhouette = silhouette_score(tsne_results, binary_labels)
    
    # Calculate centroids of each class
    failed_mask = np.array(binary_labels) == 0
    success_mask = ~failed_mask

    if np.sum(failed_mask) == 0 or np.sum(success_mask) == 0:
        return -float('inf')
    
    failed_centroid = np.mean(tsne_results[failed_mask], axis=0)
    success_centroid = np.mean(tsne_results[success_mask], axis=0)
    
    # Calculate distance between centroids
    centroid_dist = np.linalg.norm(failed_centroid - success_centroid)
    
    # Calculate cohesion within failed samples
    if len(tsne_results[failed_mask]) > 1:
        failed_cohesion = np.mean(pairwise_distances(tsne_results[failed_mask]))
    else:
        failed_cohesion = 0.1  # Small value to avoid division by zero
        
    # Combine metrics (higher is better)
    separation_score = silhouette + (centroid_dist / failed_cohesion)
    
    return separation_score

def evaluate_params(params):
    """Helper function for t-SNE parameter evaluation"""
    perp, lr, max_iter, early_exag, metric, encodings, binary_labels, overpotentials = params
    try:
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            learning_rate=lr,
            n_iter=max_iter,
            early_exaggeration=early_exag,
            metric=metric,
            random_state=42
        )
        tsne_results = tsne.fit_transform(encodings)
        score = calculate_separation_score(tsne_results, binary_labels, overpotentials)
        return (score, perp, lr, max_iter, early_exag, metric, tsne_results)
    except Exception as e:
        # Handle exceptions gracefully
        return (-float('inf'), perp, lr, max_iter, early_exag, metric, None)

def plot_tsne(encoded_samples, preset_params=None):
    """
    Plots a 2D t-SNE transformation of the encoded samples and calculates separation metrics.
    
    Parameters:
    - encoded_samples: List of tuples containing encoded input DataFrame and sample name
    """
    # Prepare data
    encodings = []
    overpotentials = []
    category_labels = []
    origins = []
    for encoded_input, _ in encoded_samples:
        encodings.append(encoded_input.drop(columns=['Sample Name', 'Category_Label', 'Overpotential mV @10 mA cm-2', 'Origin']).values.flatten())
        overpotentials.append(encoded_input['Overpotential mV @10 mA cm-2'].values[0])
        category_labels.append(encoded_input['Category_Label'].values[0])
        origins.append(encoded_input['Origin'].values[0])
    
    encodings = np.array(encodings)
    
    # Apply feature scaling
    scaler = StandardScaler()
    encodings_scaled = scaler.fit_transform(encodings)
    
    # Convert 'Null' strings to NaN for proper handling
    overpotentials = [np.nan if val == 'Null' else float(val) for val in overpotentials]
    
    # Create binary labels for failed vs non-failed samples
    binary_labels = [0 if pd.isnull(val) else 1 for val in overpotentials]
    print("Unique category labels:", set(category_labels))
    if preset_params:
        print("Using preset t-SNE parameters...")
        best_params = preset_params
    else:
        print("Using default t-SNE parameters...")
        best_params = {
            'perplexity': 75,
            'learning_rate': 5000,
            'max_iter': 4250,
            'early_exaggeration': 42,
            'metric': 'manhattan'
        }
    
    best_result = evaluate_params((
        best_params['perplexity'],
        best_params['learning_rate'],
        best_params['max_iter'],
        best_params['early_exaggeration'],
        best_params['metric'],
        encodings_scaled,
        binary_labels,
        overpotentials
    ))
    
    best_score = best_result[0]
    best_results = best_result[6]
    
    if best_score > -float('inf'):
        print("\nFinal best configuration:")
        print(f"Perplexity: {best_params['perplexity']}")
        print(f"Learning rate: {best_params['learning_rate']}")
        print(f"max_iter: {best_params['max_iter']}")
        print(f"Early exaggeration: {best_params['early_exaggeration']}")
        print(f"Metric: {best_params['metric']}")
        print(f"Combined Score: {best_score:.3f}")
        print("\nScore interpretation:")
        print("  Higher scores indicate better separation between failed and successful samples")
        print("  and better clustering of failed samples")
        
        # Plot with best configuration
        plt.figure(figsize=(10, 8))
        
        def get_marker(category):
            if 'best' in category:
                return 'o'
            elif 'variance' in category:
                return 's'
            else:
                return 'o'  # Default marker

        # Define colors for each batch
        batch_colors = {
            '1st': (0.87, 0.18, 0.54, 0.5),  # pinkish-magenta
            '2nd': (0.10, 0.59, 0.87, 0.5),  # light blue
            '3rd': (0.29, 0.73, 0.33, 0.5),  # bright green
            '4th': (0.87, 0.49, 0.15, 0.5),  # orange
            '5th': (0.58, 0.40, 0.74, 0.5)   # purple
        }

        # Plot failed samples first (hollow markers with grey edges)
        for i, (x, y) in enumerate(best_results):
            is_failed = binary_labels[i] == 0
            marker = get_marker(category_labels[i])
            plt.scatter(x, y, edgecolor='grey', facecolor='none' if is_failed else 'blue', marker=marker, s=100)

        # Then plot successful samples with color gradient
        for i, (x, y) in enumerate(best_results):
            if binary_labels[i] == 1:
                color = plt.cm.viridis(float(overpotentials[i]) / max([val for val in overpotentials if not pd.isnull(val)]))
                marker = get_marker(category_labels[i])
                plt.scatter(x, y, color=color, marker=marker, s=100)
        
        # Add a color bar with original overpotential values
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min([val for val in overpotentials if not pd.isnull(val)]), vmax=max([val for val in overpotentials if not pd.isnull(val)])))
        sm.set_array(overpotentials)
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Overpotential mV @10 mA cm-2')
        
        plt.title('t-SNE of Encoded Samples')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        # Plot convex hulls for each batch
        for origin in set(origins):
            points = np.array([best_results[i] for i in range(len(best_results)) if origins[i] == origin])
            if len(points) > 2:  # Convex hull requires at least 3 points
                hull = ConvexHull(points)
                polygon = Polygon(points[hull.vertices], closed=True, facecolor=batch_colors[origin], edgecolor='none')
                plt.gca().add_patch(polygon)
        
        plt.tight_layout()
        plt.savefig('overall_tsne.png')
        plt.close()
        
        # Plot separate t-SNE figures for each origin
        for origin in set(origins):
            plt.figure(figsize=(10, 8))
            for i, (x, y) in enumerate(best_results):
                if origins[i] == origin:
                    is_failed = binary_labels[i] == 0
                    marker = get_marker(category_labels[i])
                    if is_failed:
                        plt.scatter(x, y, edgecolor='grey', facecolor='none', marker=marker, s=100)
                    else:
                        color = plt.cm.viridis(float(overpotentials[i]) / max([val for val in overpotentials if not pd.isnull(val)]))
                        plt.scatter(x, y, color=color, marker=marker, s=100)
            
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min([val for val in overpotentials if not pd.isnull(val)]), vmax=max([val for val in overpotentials if not pd.isnull(val)])))
            sm.set_array(overpotentials)
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label('Overpotential mV @10 mA cm-2')
            
            plt.title(f't-SNE of Encoded Samples from {origin} Batch')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            
            # Plot convex hull for the current batch
            points = np.array([best_results[i] for i in range(len(best_results)) if origins[i] == origin])
            if len(points) > 2:  # Convex hull requires at least 3 points
                hull = ConvexHull(points)
                polygon = Polygon(points[hull.vertices], closed=True, facecolor=batch_colors[origin], edgecolor='none')
                plt.gca().add_patch(polygon)
            
            plt.tight_layout()
            plt.savefig(f'{origin}_tsne.png')
            plt.close()
            
            # Calculate and print silhouette scores for each batch
            if origin in ['1st', '2nd', '3rd']:
                failed_mask = np.array(binary_labels) == 0
                success_mask = ~failed_mask
                batch_mask = np.array(origins) == origin
                if np.sum(failed_mask & batch_mask) > 1 and np.sum(success_mask & batch_mask) > 1:
                    silhouette_failed_vs_success = silhouette_score(best_results[batch_mask], np.array(binary_labels)[batch_mask])
                    print(f"Silhouette score of failed vs succeeded samples in {origin} batch: {silhouette_failed_vs_success:.3f}")
                if np.sum(batch_mask) > 1:
                    silhouette_lowest_vs_highest = silhouette_score(best_results[batch_mask], np.array(category_labels)[batch_mask])
                    print(f"Silhouette score of lowest potential vs highest variance samples in {origin} batch: {silhouette_lowest_vs_highest:.3f}")
            elif origin == '4th':
                batch_mask = np.array(origins) == origin
                if np.sum(batch_mask) > 1:
                    silhouette_lowest_vs_highest = silhouette_score(best_results[batch_mask], np.array(category_labels)[batch_mask])
                    print(f"Silhouette score of lowest potential vs highest variance samples in {origin} batch: {silhouette_lowest_vs_highest:.3f}")
            elif origin == '5th':
                continue  # No need to print for 5th batch

        # Calculate and print silhouette score for overall t-SNE
        failed_mask = np.array(binary_labels) == 0
        batch_5_mask = np.array(origins) == '5th'
        if np.sum(failed_mask) > 1 and np.sum(batch_5_mask) > 1:
            silhouette_batch5_vs_failed = silhouette_score(best_results, np.array([1 if origin == '5th' else 0 for origin in origins]))
            print(f"Silhouette score of batch 5 samples vs all failed samples of all batches: {silhouette_batch5_vs_failed:.3f}")
    else:
        print("No valid configuration found. Please check your data and parameters.")

if __name__ == '__main__':
    # Example usage:
    encoded_samples = load_encoded_csv()
    preset_params = {
        'perplexity': 75,
        'learning_rate': 5000,
        'max_iter': 4250,
        'early_exaggeration': 42,
        'metric': 'manhattan'
    }

    plot_tsne(encoded_samples,preset_params=preset_params)