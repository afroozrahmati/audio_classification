

configuartion = {
    "client_counts": "2",
    "gamma": "1",
    "local_epochs": "5",
    "server_rounds": "10",
    "batch_size" :"64",
    "features" : "40" ,
    "timesteps" :"128",
    "dataset" :"physionet",
    "min_eval_clients" : "2",
    "min_fit_clients":"2" ,
    "fraction_fit":"0.3",
    "fraction_eval":"0.2",
    "evaluation_type":"server", #client or server : we want to get the result from server or client
    "server_evaluation_dataset":"physionet",
    "client_evaluation_dataset":"pascal"
}
