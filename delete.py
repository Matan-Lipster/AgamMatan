import pickle

#file_path = r"F:\HCP_resting_state_base_norm\LH_Default_PFC_1.pkl"
file_path = r"F:\HCP_DATA\100610\LH_Default_PFC_1.pkl"
#file_path = r"D:\Final Project\Predicting Human Brain States with Transformer\Gal&Yuval code\Processed_Matrices\100610\movie_1.pkl"
#file_path = r"F:\HCP_movie_base_norm\LH_Default_PFC_1.pkl"
try:
    with open(file_path, 'rb') as file:
        data = pickle.load(file)

    print("Data shape:", data.shape)
    print("Data type:", type(data))

    # Show example row
    print("\nExample row:")
    print(data.iloc[1, -5:])

    # Count how many times each y value appears (overall)
    print("\nCounts of each y value (all subjects):")
    print(data['y'].value_counts().sort_index())

    # If you want per subject AND per y:
    print("\nCounts of each y value per subject:")
    print(data.groupby(['Subject', 'y']).size())

except FileNotFoundError:
    print(f"File not found: {file_path}")
except pickle.UnpicklingError:
    print("Error: The file content is not a valid pickle format.")
except EOFError:
    print("Error: The file is incomplete or corrupted.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")