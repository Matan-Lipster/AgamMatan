# AgamMatan
Agam Valdman and Matan Lipster's final project

In general we have three major files that together build our project, within our Git you can see all of their slight variations as we have expiramented with some different models and datasets. The three main parts are the main, the precprocess and the model. Besides that we also have a file named: "delete.py" that we use to print a specific fMRI pkl file.
As previously mentiond we have tested some major changes to our code, those changes can be summarized by: 1)our original transformer code and dataset, 2) using our colleges' dataset and transformer model (fiels with gy_) and 3) using a PLS model with various datasets and options (files with pls_).

Manual:
Using our regular transformer workflow- 
1)Go to the file named "main_try.py", in there you'll be able to change all the parameters of the transformer and the dataset, whilst there are a lot of editable parameters, the most unintuitive are the: NET_list, NET_indexes and the H_list. these 3 parameters are responsible for the sub-network to be chosen, the specific ROIs (reigon of interest) to be assesd of those sub-networks and the brain hemosphere to focus on (left and/or right). in each of these parameters you can choose as many ROIs/networks/hemospheres as you'd like!
2) Run the file named "main_try.py"  with the parameters of your choosing
3) press 1 when asked to choose one of the options, this action will cause the test results to be the fluid intelligence unique features

Using our colleges transformer model and dataset- 
1)Go to the file named "gy_main.py", in there you'll be able to change all the parameters of the transformer and the dataset, whilst there are a lot of editable parameters, the most unintuitive are the: NET_list, NET_indexes and the H_list. these 3 parameters are responsible for the sub-network to be chosen, the specific ROIs (reigon of interest) to be assesd of those sub-networks and the brain hemosphere to focus on (left and/or right). in each of these parameters you can choose as many ROIs/networks/hemospheres as you'd like!
2) Run the file named "gy_main.py"  with the parameters of your choosing
3) press 1 when asked to choose one of the options, this action will cause the test results to be the fluid intelligence unique features
*note that this workflow combines all of the selected ROIs into one matix and due to that forces an average on the data.

Using our PLS model workflow- 
1)Go to the file named "pls_main.py",in here there are a lot of different options for both the test features and the test data. the features are: fluid intelligence/personality/ fluid intelligence 2/ all with fluid intelligence 2 being two out of the 3 intelligence features and all is simply all posible features and the data is chosen by- 1:run on ROIs, 2:run on sub-networks, 3:run on full networks, 300: run on all 300 regions, rest: resting state data all regions, movies: run on per movie data, and all: which is all networks all features. besides that there are also different possible datasets that you can assign to the directory parameter, we have the regular dataset: r"F:\HCP_DATA" and from there on you can uncomment any path looking line and use a different one.
* note that all of the options that uses a structure bigger then an ROI (i.e networks, subnetworks and movies) will apply an average on the matrices, there is also the option to instead apply either PCA or MEAN. theres also an option to apply 100 permutation by setting MUTATE to True.
