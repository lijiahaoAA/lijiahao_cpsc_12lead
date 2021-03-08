class Config():

    train_mat_path = 'D:/data-set/12leadsECG/TrainingSet/'
    test_mat_path = 'D:/data-set/12leadsECG/validation_set/'
    all_train = 'ECG_train_data_process_3600QRS.npy'
    all_test_300 = 'ECG_test_data_process_3600QRS_300record_new.npy'
    all_test_500 = 'ECG_test_data_process_3600QRS_500record.npy'

    epoch = 60
    momentum = 0.9
    sample_frequency = 500
    normalization_min = -3
    normalization_max = 3
    max_data = 17.881176
    min_data = -14.625054