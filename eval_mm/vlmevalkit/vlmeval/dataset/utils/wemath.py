# pylint: skip-file

import pandas as pd
import json
import numpy as np
import os
import argparse

# four_dimensional_metrics.py


# Function to evaluate steps
def evaluate_evaluate_steps(json, steps):  # noqa
    jokers = [json[[f'joker_{i}', f'knowledge concept_{i}']] for i in range(1, steps + 1)]
    for i in range(steps):
        jokers[i].rename(
            columns={f'joker_{i + 1}': 'joker', f'knowledge concept_{i + 1}': 'knowledge_concept'},
            inplace=True,
        )
    concatenated_steps = pd.concat(jokers, axis=0)
    return concatenated_steps


# Function to load and process JSON data
def load_and_process_data(filepath):
    df = pd.read_excel(filepath)
    if 'hit' not in df.columns:
        df['processed_answer'] = (
            df['prediction']
            .str.split('Answer')
            .str[-1]
            .str.strip()
            .str.replace(r'[>><<:.]', '', regex=True)
            .str.strip()
        )
        df['processed_answer'] = df['processed_answer'].apply(lambda x: x[0] if x and x[0] in 'ABCDEFGH' else None)
        df['joker'] = df['processed_answer'] == df['answer']
    else:
        df['joker'] = df['hit'].astype(bool)
    return df


# Function to process steps data and merge results
def evaluate_process_steps_data(df, steps):
    steps_data = {f'{steps}steps_{i}': df[df['key'] == f'{steps}steps_{i}'] for i in range(1, steps + 1)}
    steps_data[f'{steps}steps_multi'] = df[df['key'] == f'{steps}steps_multi']
    for key, data in steps_data.items():
        data.columns = [col + f'_{key.split("_")[-1]}' for col in data.columns]
    merged_data = steps_data[f'{steps}steps_1']
    for i in range(2, steps + 1):
        merged_data = pd.merge(
            merged_data, steps_data[f'{steps}steps_{i}'], left_on=f'ID_1', right_on=f'ID_{i}', how='left' # noqa
        )
    merged_data = pd.merge(
        merged_data, steps_data[f'{steps}steps_multi'], left_on=f'ID_1', right_on='ID_multi', how='left'  # noqa
    )
    return merged_data


# Function to calculate evaluation metrics
def evaluate_calculate_metrics(merged_2steps, merged_3steps):
    metrics = {}
    metrics['steps2_filtered_rows_1_loose'] = merged_2steps[
        ((merged_2steps['joker_1'] == False) & (merged_2steps['joker_2'] == False))  # noqa
        & (merged_2steps['joker_multi'] == True)  # noqa
    ]
    metrics['steps2_filtered_rows_1_strict'] = merged_2steps[
        ((merged_2steps['joker_1'] == False) | (merged_2steps['joker_2'] == False))  # noqa
        & (merged_2steps['joker_multi'] == True)  # noqa
    ]
    metrics['steps2_filtered_rows_2'] = merged_2steps[
        ((merged_2steps['joker_1'] == True) & (merged_2steps['joker_2'] == True))  # noqa
        & (merged_2steps['joker_multi'] == False)  # noqa
    ]
    metrics['steps2_filtered_rows_3'] = merged_2steps[
        ((merged_2steps['joker_1'] == False) | (merged_2steps['joker_2'] == False))  # noqa
        & (merged_2steps['joker_multi'] == False)  # noqa
    ]
    metrics['steps2_filtered_rows_4_loose'] = merged_2steps[
        ((merged_2steps['joker_1'] == True) | (merged_2steps['joker_2'] == True))
        & (merged_2steps['joker_multi'] == True)
    ]
    metrics['steps2_filtered_rows_4_strict'] = merged_2steps[
        ((merged_2steps['joker_1'] == True) & (merged_2steps['joker_2'] == True))
        & (merged_2steps['joker_multi'] == True)
    ]
    metrics['steps3_filtered_rows_1_loose'] = merged_3steps[
        (
            (merged_3steps['joker_1'] == False)
            & (merged_3steps['joker_2'] == False)
            & (merged_3steps['joker_3'] == False)
        )
        & (merged_3steps['joker_multi'] == True)
    ]
    metrics['steps3_filtered_rows_1_strict'] = merged_3steps[
        (
            (merged_3steps['joker_1'] == False)
            | (merged_3steps['joker_2'] == False)
            | (merged_3steps['joker_3'] == False)
        )
        & (merged_3steps['joker_multi'] == True)
    ]
    metrics['steps3_filtered_rows_2'] = merged_3steps[
        ((merged_3steps['joker_1'] == True) & (merged_3steps['joker_2'] == True) & (merged_3steps['joker_3'] == True))
        & (merged_3steps['joker_multi'] == False)
    ]
    metrics['steps3_filtered_rows_3'] = merged_3steps[
        (
            (merged_3steps['joker_1'] == False)
            | (merged_3steps['joker_2'] == False)
            | (merged_3steps['joker_3'] == False)
        )
        & (merged_3steps['joker_multi'] == False)
    ]
    metrics['steps3_filtered_rows_4_loose'] = merged_3steps[
        ((merged_3steps['joker_1'] == True) | (merged_3steps['joker_2'] == True) | (merged_3steps['joker_3'] == True))
        & (merged_3steps['joker_multi'] == True)
    ]
    metrics['steps3_filtered_rows_4_strict'] = merged_3steps[
        ((merged_3steps['joker_1'] == True) & (merged_3steps['joker_2'] == True) & (merged_3steps['joker_3'] == True))
        & (merged_3steps['joker_multi'] == True)
    ]
    # metrics.to_csv("/Users/mac/Desktop/测试结果/error_anal/csv/gpt4o-0626.csv", index = False)
    return metrics


# Function to compute evaluation rates and final scores
def evaluate_compute_final_scores(metrics, total_count):
    total_counts = {
        'InadequateGeneralization': len(metrics['steps2_filtered_rows_2']) + len(metrics['steps3_filtered_rows_2']),
        'InsufficientKnowledge': len(metrics['steps2_filtered_rows_3']) + len(metrics['steps3_filtered_rows_3']),
        'CompleteMastery_loose': len(metrics['steps2_filtered_rows_4_loose'])
        + len(metrics['steps3_filtered_rows_4_loose']),
        'CompleteMastery_strict': len(metrics['steps2_filtered_rows_4_strict'])
        + len(metrics['steps3_filtered_rows_4_strict']),
        'RoteMemorization_loose': len(metrics['steps2_filtered_rows_1_loose'])
        + len(metrics['steps3_filtered_rows_1_loose']),
        'RoteMemorization_strict': len(metrics['steps2_filtered_rows_1_strict'])
        + len(metrics['steps3_filtered_rows_1_strict']),
    }
    rates = {
        'InadequateGeneralization_rate': "{:.2%}".format(total_counts['InadequateGeneralization'] / total_count),
        'InsufficientKnowledge_rate': "{:.2%}".format(total_counts['InsufficientKnowledge'] / total_count),
        'CompleteMastery_loose_rate': "{:.2%}".format(total_counts['CompleteMastery_loose'] / total_count),
        'CompleteMastery_strict_rate': "{:.2%}".format(total_counts['CompleteMastery_strict'] / total_count),
        'RoteMemorization_loose_rate': "{:.2%}".format(
            total_counts['RoteMemorization_loose']
            / (total_counts['CompleteMastery_loose'] + total_counts['RoteMemorization_loose'])
        ),
        'RoteMemorization_strict_rate': "{:.2%}".format(
            total_counts['RoteMemorization_strict']
            / (total_counts['CompleteMastery_strict'] + total_counts['RoteMemorization_strict'])
        ),
    }
    return total_counts, rates


# Function to update main results DataFrame
def evaluate_update_main_results_df(main_results_df, total_counts, rates):

    final_score_loose = "{:.2%}".format(
        (
            525
            - 0.5 * total_counts['InadequateGeneralization']
            - total_counts['RoteMemorization_loose']
            - total_counts['InsufficientKnowledge']
        )
        / 525
    )
    final_score_strict = "{:.2%}".format(
        (
            525
            - 0.5 * total_counts['InadequateGeneralization']
            - total_counts['RoteMemorization_strict']
            - total_counts['InsufficientKnowledge']
        )
        / 525
    )

    new_row = {
        # 'Model': model,
        'Score (Strict)': final_score_strict,
        'InsufficientKnowledge (Strict)': f"{rates['InsufficientKnowledge_rate']} ({total_counts['InsufficientKnowledge']})",
        'InadequateGeneralization (Strict)': f"{rates['InadequateGeneralization_rate']} ({total_counts['InadequateGeneralization']})",
        'CompleteMastery (Strict)': f"{rates['CompleteMastery_strict_rate']} ({total_counts['CompleteMastery_strict']})",
        'RoteMemorization (Strict)': f"{rates['RoteMemorization_strict_rate']} ({total_counts['RoteMemorization_strict']})",
        'Score (Loose)': final_score_loose,
        'InsufficientKnowledge (Loose)': f"{rates['InsufficientKnowledge_rate']} ({total_counts['InsufficientKnowledge']})",
        'InadequateGeneralization (Loose)': f"{rates['InadequateGeneralization_rate']} ({total_counts['InadequateGeneralization']})",
        'CompleteMastery (Loose)': f"{rates['CompleteMastery_loose_rate']} ({total_counts['CompleteMastery_loose']})",
        'RoteMemorization (Loose)': f"{rates['RoteMemorization_loose_rate']} ({total_counts['RoteMemorization_loose']})",
    }
    main_results_df = main_results_df._append(new_row, ignore_index=True)
    return main_results_df


# Main function to evaluate models
def wemath_evaluate_models(output_json, main_results_csv_path=None):

    main_results_df = pd.DataFrame(
        columns=[
            'Model',
            'Score (Strict)',
            'InsufficientKnowledge (Strict)',
            'InadequateGeneralization (Strict)',
            'CompleteMastery (Strict)',
            'RoteMemorization (Strict)',
            'Score (Loose)',
            'InsufficientKnowledge (Loose)',
            'InadequateGeneralization (Loose)',
            'CompleteMastery (Loose)',
            'RoteMemorization (Loose)',
        ]
    )

    # print(f"Evaluating model: {model_name}, JSON path: {output_json}")
    data = load_and_process_data(output_json)
    data_2steps = data[data['key'].str.contains('2steps')]
    data_3steps = data[data['key'].str.contains('3steps')]
    merged_2steps = evaluate_process_steps_data(data_2steps, 2)
    merged_3steps = evaluate_process_steps_data(data_3steps, 3)

    metrics = evaluate_calculate_metrics(merged_2steps, merged_3steps)
    total_counts, rates = evaluate_compute_final_scores(metrics, total_count=525)

    main_results_df = evaluate_update_main_results_df(main_results_df, total_counts, rates)

    print(main_results_df.to_string(index=False))
    if main_results_csv_path is not None:
        main_results_df.to_csv(main_results_csv_path, index=False)
        print("Evaluation completed and results saved to CSV.")
    return main_results_df.to_dict()


### Accuracy.py
# Function to load knowledge structure nodes
def load_knowledge_structure_nodes(filepath):
    # with open(filepath, "r") as file:
    #     nodes = json.load(file)
    nodes = knowledge_structure_nodes
    nodes = pd.DataFrame(nodes)
    nodes['final_key'] = nodes['full node'].str.split('_').str[-1]
    nodes['root_2'] = nodes['full node'].str.split('_').str[1]
    return nodes


# Function to evaluate steps
def accuracy_evaluate_steps(json, steps, nodes):
    jokers = [json[[f'joker_{i}', f'knowledge concept_{i}']] for i in range(1, steps + 1)]
    for i in range(steps):
        jokers[i] = pd.merge(
            jokers[i],
            nodes[['final_key', 'full node', 'root_2']],
            left_on=f'knowledge concept_{i + 1}',
            right_on='final_key',
            how='left',
        )
        jokers[i].rename(
            columns={f'joker_{i + 1}': 'joker', f'knowledge concept_{i + 1}': 'knowledge_concept'},
            inplace=True,
        )
    concatenated_steps = pd.concat(jokers, axis=0)
    return concatenated_steps


# Function to process steps data and merge results
def accuracy_process_steps_data(df, steps):
    steps_data = {f'{steps}steps_{i}': df[df['key'] == f'{steps}steps_{i}'] for i in range(1, steps + 1)}
    steps_data[f'{steps}steps_multi'] = df[df['key'] == f'{steps}steps_multi']
    for key, data in steps_data.items():
        data.columns = [col + f'_{key.split("_")[-1]}' for col in data.columns]
    merged_data = steps_data[f'{steps}steps_1']
    for i in range(2, steps + 1):
        merged_data = pd.merge(
            merged_data, steps_data[f'{steps}steps_{i}'], left_on=f'ID_1', right_on=f'ID_{i}', how='left'
        )
    merged_data = pd.merge(
        merged_data, steps_data[f'{steps}steps_multi'], left_on=f'ID_1', right_on='ID_multi', how='left'
    )
    return merged_data


# Function to update main results DataFrame
def accuracy_update_main_results_df(nodes, main_results_df, concatenated_data, merged_2steps, merged_3steps):
    One_step_acc = "{:.2%}".format(concatenated_data['joker'].mean())
    Two_step_acc = "{:.2%}".format(merged_2steps['joker_multi'].mean())
    Three_step_acc = "{:.2%}".format(merged_3steps['joker_multi'].mean())

    new_row = {
        # 'Model': model_name,
        'One-step(S1)': One_step_acc,
        'Two-step(S2)': Two_step_acc,
        'Three-step(S3)': Three_step_acc,
    }
    # Calculate rates according to Nodes
    nodes['final_rode'] = nodes['full node'].str.split('_').str[-1]
    csv_final_score = concatenated_data.groupby('final_key')['joker'].mean()
    csv_final_score = pd.merge(nodes, csv_final_score, left_on='final_rode', right_on='final_key', how='left')

    new_row.update(csv_final_score.groupby('root2')['joker'].mean().apply(lambda x: "{:.2%}".format(x)).to_dict())
    main_results_df = main_results_df._append(new_row, ignore_index=True)

    return main_results_df


# Main function to evaluate models
def wemath_accuracy(output_json, main_results_csv_path=None):

    # nodes = load_knowledge_structure_nodes(knowledge_structure_nodes_path)
    nodes = knowledge_structure_nodes
    nodes = pd.DataFrame(nodes)
    nodes['final_key'] = nodes['full node'].str.split('_').str[-1]
    nodes['root_2'] = nodes['full node'].str.split('_').str[1]

    main_results_df = pd.DataFrame(
        columns=[
            'Model',
            'One-step(S1)',
            'Two-step(S2)',
            'Three-step(S3)',
            'Understanding and Conversion of Units',
            'Angles and Length',
            'Calculation of Plane Figures',
            'Understanding of Plane Figures',
            'Calculation of Solid Figures',
            'Understanding of Solid Figures',
            'Basic Transformations of Figures',
            'Cutting and Combining of Figures',
            'Direction',
            'Position',
            'Route Map',
            'Correspondence of Coordinates and Positions',
        ]
    )

    # print(f"Evaluating model: {model_name}, JSON path: {output_json}")
    data = load_and_process_data(output_json)
    data_2steps = data[data['key'].str.contains('2steps')]
    data_3steps = data[data['key'].str.contains('3steps')]
    merged_2steps = accuracy_process_steps_data(data_2steps, 2)
    merged_3steps = accuracy_process_steps_data(data_3steps, 3)

    concatenated_data = pd.concat(
        [accuracy_evaluate_steps(merged_2steps, 2, nodes), accuracy_evaluate_steps(merged_3steps, 3, nodes)],
        axis=0,
    )
    main_results_df = accuracy_update_main_results_df(
        nodes, main_results_df, concatenated_data, merged_2steps, merged_3steps
    )

    print(main_results_df.to_string(index=False))
    if main_results_csv_path is not None:
        main_results_df.to_csv(main_results_csv_path, index=False)
        print("Evaluation completed and results saved to CSV.")

    return main_results_df.to_dict()


knowledge_structure_nodes = [
    {
        "root0": "Geometry and Figures",
        "root1": "Measurement",
        "root2": "Understanding and Conversion of Units",
        "root3": "Conversion Rates and Calculations Between Area Units",
        "root4": None,
        "full node": "Measurement_Understanding and Conversion of Units_Conversion Rates and Calculations Between Area Units",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Measurement",
        "root2": "Understanding and Conversion of Units",
        "root3": "Conversion Rates and Calculations Between Volume Units (Including Liters and Milliliters)",
        "root4": None,
        "full node": "Measurement_Understanding and Conversion of Units_Conversion Rates and Calculations Between Volume Units (Including Liters and Milliliters)",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Measurement",
        "root2": "Understanding and Conversion of Units",
        "root3": "Conversion Rates and Calculations Between Length Units",
        "root4": None,
        "full node": "Measurement_Understanding and Conversion of Units_Conversion Rates and Calculations Between Length Units",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Measurement",
        "root2": "Angles and Length",
        "root3": "Understanding Angles (Using a Protractor)",
        "root4": None,
        "full node": "Measurement_Angles and Length_Understanding Angles (Using a Protractor)",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Measurement",
        "root2": "Angles and Length",
        "root3": "Understanding Length (Using a Ruler)",
        "root4": None,
        "full node": "Measurement_Angles and Length_Understanding Length (Using a Ruler)",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Calculation of Solid Figures",
        "root3": "Calculation of Surface Area of Solid Figures",
        "root4": "Surface Area of Cylinders",
        "full node": "Solid Figures_Calculation of Solid Figures_Calculation of Surface Area of Solid Figures_Surface Area of Cylinders",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Calculation of Solid Figures",
        "root3": "Calculation of Surface Area of Solid Figures",
        "root4": "Surface Area of Rectangular Cuboids",
        "full node": "Solid Figures_Calculation of Solid Figures_Calculation of Surface Area of Solid Figures_Surface Area of Rectangular Cuboids",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Calculation of Solid Figures",
        "root3": "Calculation of Surface Area of Solid Figures",
        "root4": "Surface Area of Cubes",
        "full node": "Solid Figures_Calculation of Solid Figures_Calculation of Surface Area of Solid Figures_Surface Area of Cubes",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Calculation of Solid Figures",
        "root3": "Calculation of Volume of Solid Figures",
        "root4": "Volume and Capacity of Cylinders",
        "full node": "Solid Figures_Calculation of Solid Figures_Calculation of Volume of Solid Figures_Volume and Capacity of Cylinders",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Calculation of Solid Figures",
        "root3": "Calculation of Volume of Solid Figures",
        "root4": "Volume and Capacity of Cones",
        "full node": "Solid Figures_Calculation of Solid Figures_Calculation of Volume of Solid Figures_Volume and Capacity of Cones",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Calculation of Solid Figures",
        "root3": "Calculation of Volume of Solid Figures",
        "root4": "Volume and Capacity of Rectangular Cuboids",
        "full node": "Solid Figures_Calculation of Solid Figures_Calculation of Volume of Solid Figures_Volume and Capacity of Rectangular Cuboids",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Calculation of Solid Figures",
        "root3": "Calculation of Volume of Solid Figures",
        "root4": "Volume and Capacity of Cubes",
        "full node": "Solid Figures_Calculation of Solid Figures_Calculation of Volume of Solid Figures_Volume and Capacity of Cubes",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Understanding of Solid Figures",
        "root3": "Expanded View of Solids",
        "root4": "Expanded View of Cylinders",
        "full node": "Solid Figures_Understanding of Solid Figures_Expanded View of Solids_Expanded View of Cylinders",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Understanding of Solid Figures",
        "root3": "Expanded View of Solids",
        "root4": "Expanded View of Rectangular Cuboids",
        "full node": "Solid Figures_Understanding of Solid Figures_Expanded View of Solids_Expanded View of Rectangular Cuboids",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Understanding of Solid Figures",
        "root3": "Expanded View of Solids",
        "root4": "Expanded View of Cubes",
        "full node": "Solid Figures_Understanding of Solid Figures_Expanded View of Solids_Expanded View of Cubes",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Understanding of Solid Figures",
        "root3": "Cylinders and Cones",
        "root4": "Properties of Cylinders",
        "full node": "Solid Figures_Understanding of Solid Figures_Cylinders and Cones_Properties of Cylinders",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Understanding of Solid Figures",
        "root3": "Cylinders and Cones",
        "root4": "Properties of Cones",
        "full node": "Solid Figures_Understanding of Solid Figures_Cylinders and Cones_Properties of Cones",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Understanding of Solid Figures",
        "root3": "Rectangular Cuboids and Cubes",
        "root4": "Properties and Understanding of Rectangular Cuboids",
        "full node": "Solid Figures_Understanding of Solid Figures_Rectangular Cuboids and Cubes_Properties and Understanding of Rectangular Cuboids",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Understanding of Solid Figures",
        "root3": "Rectangular Cuboids and Cubes",
        "root4": "Properties and Understanding of Cubes",
        "full node": "Solid Figures_Understanding of Solid Figures_Rectangular Cuboids and Cubes_Properties and Understanding of Cubes",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Solid Figures",
        "root2": "Understanding of Solid Figures",
        "root3": "Observing Objects",
        "root4": None,
        "full node": "Solid Figures_Understanding of Solid Figures_Observing Objects",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Sum of Interior Angles of Polygons",
        "root4": "Sum of Interior Angles of Other Polygons",
        "full node": "Plane Figures_Calculation of Plane Figures_Sum of Interior Angles of Polygons_Sum of Interior Angles of Other Polygons",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Sum of Interior Angles of Polygons",
        "root4": "Sum of Interior Angles of Triangles",
        "full node": "Plane Figures_Calculation of Plane Figures_Sum of Interior Angles of Polygons_Sum of Interior Angles of Triangles",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Calculation and Comparison of Angles",
        "root4": None,
        "full node": "Plane Figures_Calculation of Plane Figures_Calculation and Comparison of Angles",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Calculation of Areas",
        "root4": "Area of Parallelograms",
        "full node": "Plane Figures_Calculation of Plane Figures_Calculation of Areas_Area of Parallelograms",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Calculation of Areas",
        "root4": "Area of Triangles",
        "full node": "Plane Figures_Calculation of Plane Figures_Calculation of Areas_Area of Triangles",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Calculation of Areas",
        "root4": "Area of Sectors",
        "full node": "Plane Figures_Calculation of Plane Figures_Calculation of Areas_Area of Sectors",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Calculation of Areas",
        "root4": "Area of Trapezoids",
        "full node": "Plane Figures_Calculation of Plane Figures_Calculation of Areas_Area of Trapezoids",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Calculation of Areas",
        "root4": "Area of Circles",
        "full node": "Plane Figures_Calculation of Plane Figures_Calculation of Areas_Area of Circles",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Calculation of Areas",
        "root4": "Area of Rectangles",
        "full node": "Plane Figures_Calculation of Plane Figures_Calculation of Areas_Area of Rectangles",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Calculation of Areas",
        "root4": "Area of Squares",
        "full node": "Plane Figures_Calculation of Plane Figures_Calculation of Areas_Area of Squares",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Calculation of Perimeters",
        "root4": "Perimeter of Parallelograms",
        "full node": "Plane Figures_Calculation of Plane Figures_Calculation of Perimeters_Perimeter of Parallelograms",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Calculation of Perimeters",
        "root4": "Perimeter of Triangles",
        "full node": "Plane Figures_Calculation of Plane Figures_Calculation of Perimeters_Perimeter of Triangles",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Calculation of Perimeters",
        "root4": "Perimeter of Trapezoids",
        "full node": "Plane Figures_Calculation of Plane Figures_Calculation of Perimeters_Perimeter of Trapezoids",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Calculation of Perimeters",
        "root4": "Circumference of Circles",
        "full node": "Plane Figures_Calculation of Plane Figures_Calculation of Perimeters_Circumference of Circles",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Calculation of Perimeters",
        "root4": "Perimeter of Rectangles",
        "full node": "Plane Figures_Calculation of Plane Figures_Calculation of Perimeters_Perimeter of Rectangles",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Calculation of Plane Figures",
        "root3": "Calculation of Perimeters",
        "root4": "Perimeter of Squares",
        "full node": "Plane Figures_Calculation of Plane Figures_Calculation of Perimeters_Perimeter of Squares",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Understanding of Plane Figures",
        "root3": "Polygons",
        "root4": "Properties and Understanding of Parallelograms",
        "full node": "Plane Figures_Understanding of Plane Figures_Polygons_Properties and Understanding of Parallelograms",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Understanding of Plane Figures",
        "root3": "Polygons",
        "root4": "Properties and Understanding of Triangles",
        "full node": "Plane Figures_Understanding of Plane Figures_Polygons_Properties and Understanding of Triangles",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Understanding of Plane Figures",
        "root3": "Polygons",
        "root4": "Properties and Understanding of Trapezoids",
        "full node": "Plane Figures_Understanding of Plane Figures_Polygons_Properties and Understanding of Trapezoids",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Understanding of Plane Figures",
        "root3": "Polygons",
        "root4": "Properties and Understanding of Rectangles",
        "full node": "Plane Figures_Understanding of Plane Figures_Polygons_Properties and Understanding of Rectangles",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Understanding of Plane Figures",
        "root3": "Polygons",
        "root4": "Properties and Understanding of Squares",
        "full node": "Plane Figures_Understanding of Plane Figures_Polygons_Properties and Understanding of Squares",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Understanding of Plane Figures",
        "root3": "Classification and Understanding of Angles",
        "root4": "Understanding Triangular Rulers",
        "full node": "Plane Figures_Understanding of Plane Figures_Classification and Understanding of Angles_Understanding Triangular Rulers",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Understanding of Plane Figures",
        "root3": "Classification and Understanding of Angles",
        "root4": "Understanding and Representing Angles",
        "full node": "Plane Figures_Understanding of Plane Figures_Classification and Understanding of Angles_Understanding and Representing Angles",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Understanding of Plane Figures",
        "root3": "Properties and Understanding of Line Segments",
        "root4": "Distance Between Two Points",
        "full node": "Plane Figures_Understanding of Plane Figures_Properties and Understanding of Line Segments_Distance Between Two Points",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Understanding of Plane Figures",
        "root3": "Properties and Understanding of Line Segments",
        "root4": "Understanding Line Segments, Lines, and Rays",
        "full node": "Plane Figures_Understanding of Plane Figures_Properties and Understanding of Line Segments_Understanding Line Segments, Lines, and Rays",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Understanding of Plane Figures",
        "root3": "Positional Relationships Between Line Segments",
        "root4": "perpendicularity",
        "full node": "Plane Figures_Understanding of Plane Figures_Positional Relationships Between Line Segments_perpendicularity",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Understanding of Plane Figures",
        "root3": "Positional Relationships Between Line Segments",
        "root4": "Parallel",
        "full node": "Plane Figures_Understanding of Plane Figures_Positional Relationships Between Line Segments_Parallel",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Understanding of Plane Figures",
        "root3": "Circles and Sectors",
        "root4": "Understanding Sectors",
        "full node": "Plane Figures_Understanding of Plane Figures_Circles and Sectors_Understanding Sectors",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Understanding of Plane Figures",
        "root3": "Circles and Sectors",
        "root4": "Understanding Circles",
        "full node": "Plane Figures_Understanding of Plane Figures_Circles and Sectors_Understanding Circles",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Plane Figures",
        "root2": "Understanding of Plane Figures",
        "root3": "Observing Figures",
        "root4": None,
        "full node": "Plane Figures_Understanding of Plane Figures_Observing Figures",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Transformation and Motion of Figures",
        "root2": "Basic Transformations of Figures",
        "root3": "Axial Symmetry",
        "root4": None,
        "full node": "Transformation and Motion of Figures_Basic Transformations of Figures_Axial Symmetry",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Transformation and Motion of Figures",
        "root2": "Basic Transformations of Figures",
        "root3": "Translation",
        "root4": None,
        "full node": "Transformation and Motion of Figures_Basic Transformations of Figures_Translation",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Transformation and Motion of Figures",
        "root2": "Basic Transformations of Figures",
        "root3": "Rotation",
        "root4": None,
        "full node": "Transformation and Motion of Figures_Basic Transformations of Figures_Rotation",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Transformation and Motion of Figures",
        "root2": "Cutting and Combining of Figures",
        "root3": "Combining and Dividing Solids",
        "root4": None,
        "full node": "Transformation and Motion of Figures_Cutting and Combining of Figures_Combining and Dividing Solids",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Transformation and Motion of Figures",
        "root2": "Cutting and Combining of Figures",
        "root3": "Combining Plane Figures",
        "root4": "Division of Plane Figures",
        "full node": "Transformation and Motion of Figures_Cutting and Combining of Figures_Combining Plane Figures_Division of Plane Figures",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Transformation and Motion of Figures",
        "root2": "Cutting and Combining of Figures",
        "root3": "Combining Plane Figures",
        "root4": "Combining Plane Figures",
        "full node": "Transformation and Motion of Figures_Cutting and Combining of Figures_Combining Plane Figures_Combining Plane Figures",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Transformation and Motion of Figures",
        "root2": "Cutting and Combining of Figures",
        "root3": "Combining Plane Figures",
        "root4": "Tessellation of Figures",
        "full node": "Transformation and Motion of Figures_Cutting and Combining of Figures_Combining Plane Figures_Tessellation of Figures",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Transformation and Motion of Figures",
        "root2": "Cutting and Combining of Figures",
        "root3": "Combining Plane Figures",
        "root4": "Folding Problems of Figures",
        "full node": "Transformation and Motion of Figures_Cutting and Combining of Figures_Combining Plane Figures_Folding Problems of Figures",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Position and Direction",
        "root2": "Direction",
        "root3": "Southeast, Southwest, Northeast, Northwest Directions",
        "root4": None,
        "full node": "Position and Direction_Direction_Southeast, Southwest, Northeast, Northwest Directions",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Position and Direction",
        "root2": "Direction",
        "root3": "Cardinal Directions (East, South, West, North)",
        "root4": None,
        "full node": "Position and Direction_Direction_Cardinal Directions (East, South, West, North)",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Position and Direction",
        "root2": "Route Map",
        "root3": "Determining the Positions of Objects Based on Direction, Angle, and Distance",
        "root4": None,
        "full node": "Position and Direction_Route Map_Determining the Positions of Objects Based on Direction, Angle, and Distance",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Position and Direction",
        "root2": "Route Map",
        "root3": "Describing Simple Routes Based on Direction and Distance",
        "root4": None,
        "full node": "Position and Direction_Route Map_Describing Simple Routes Based on Direction and Distance",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Position and Direction",
        "root2": "Correspondence of Coordinates and Positions",
        "root3": "Representing Positions Using Ordered Pairs",
        "root4": None,
        "full node": "Position and Direction_Correspondence of Coordinates and Positions_Representing Positions Using Ordered Pairs",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Position and Direction",
        "root2": "Correspondence of Coordinates and Positions",
        "root3": "Finding Positions Based on Ordered Pairs",
        "root4": None,
        "full node": "Position and Direction_Correspondence of Coordinates and Positions_Finding Positions Based on Ordered Pairs",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Position and Direction",
        "root2": "Position",
        "root3": "Front-Back Position",
        "root4": None,
        "full node": "Position and Direction_Position_Front-Back Position",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Position and Direction",
        "root2": "Position",
        "root3": "Up-Down Position",
        "root4": None,
        "full node": "Position and Direction_Position_Up-Down Position",
    },
    {
        "root0": "Geometry and Figures",
        "root1": "Position and Direction",
        "root2": "Position",
        "root3": "Left-Right Position",
        "root4": None,
        "full node": "Position and Direction_Position_Left-Right Position",
    },
]
