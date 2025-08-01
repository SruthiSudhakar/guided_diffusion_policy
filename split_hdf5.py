import h5py, re, sys
from tqdm import tqdm
# Path to the original and new HDF5 files
start=int(sys.argv[1])
end=int(sys.argv[2])
original_file_path = 'externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_im128.hdf5'
new_file_path =f'externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_im128_indices_{start}_{end-1}.hdf5'

print('saving to new path:', new_file_path)
# Open the original HDF5 file
with h5py.File(original_file_path, 'r') as original_file:
    # Create a new HDF5 file
    with h5py.File(new_file_path, 'w') as new_file:
        
        # Copy the 'data' attributes to the new file
        data_group = original_file['data']
        new_data_group = new_file.create_group('data')
        
        # Copy all attributes from 'data'
        for attr_name, attr_value in data_group.attrs.items():
            new_data_group.attrs[attr_name] = attr_value
        
        def extract_number(episode_name):
            # Use regular expression to extract the number after 'demo_'
            match = re.search(r'(\d+)', episode_name)
            return int(match.group(0))

        sorted_episode_names = sorted(list(original_file['data'].keys()), key=extract_number)

        # Loop over each episode (demo)
        for episode_name in tqdm(sorted_episode_names[start:end]):  # Only the first 128 demos
            episode_data = original_file['data'][episode_name]
            
            # Create the same structure in the new file
            new_episode_data = new_data_group.create_group(episode_name)
            
            # Copy the attributes for each demo (like 'ep_meta', 'model_file', 'num_samples')
            for attr_name, attr_value in episode_data.attrs.items():
                new_episode_data.attrs[attr_name] = attr_value
            
            # Copy all datasets under this episode
            for key in episode_data.keys():
                dataset = episode_data[key]
                
                # If it's a dataset (not a group), copy it directly
                if isinstance(dataset, h5py.Dataset):
                    new_episode_data.create_dataset(key, data=dataset[:])  # Copy dataset with slicing
                # If it's a group (like "obs"), recursively copy datasets inside
                elif isinstance(dataset, h5py.Group):
                    new_group = new_episode_data.create_group(key)
                    for subgroup_key in dataset.keys():
                        subgroup_dataset = dataset[subgroup_key]
                        if isinstance(subgroup_dataset, h5py.Dataset):  # Make sure it's a dataset
                            # Skip image data (typically has shape with 3 or 4 dimensions for images)
                            if 'image' in subgroup_key.lower() or 'img' in subgroup_key.lower() or (len(subgroup_dataset.shape) >= 3 and subgroup_dataset.shape[-1] in [1, 3, 4]):
                                print(f"Skipping image data: {episode_name}/{key}/{subgroup_key} with shape {subgroup_dataset.shape}")
                                continue
                            new_group.create_dataset(subgroup_key, data=subgroup_dataset[:])  # Use slicing to copy data
# Explicitly close the files
original_file.close()
new_file.close()
