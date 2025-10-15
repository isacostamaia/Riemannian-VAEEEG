#%%

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Sampler

import numpy as np
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split

from moabb.datasets import BI2013a, Zhou2016
from moabb.paradigms import P300, MotorImagery, LeftRightImagery
from pyriemann.spatialfilters import Xdawn
from pyriemann.estimation import ERPCovariances, Covariances

import shutil
from distutils.dir_util import copy_tree

shutil.copy_tree = copy_tree

from utils.manifold import SPDManifold

def moabb_to_epochs(dataset, paradigm, subjects=None, classes_of_int=None):
    """
    Gets dataset from moabb
        Returns X data array of shape (epochs, channels, times)
        y labels array of length epochs 
    """

    if not subjects:
        subjects = dataset.subject_list #:4
    epochs, y, meta = paradigm.get_data(dataset=dataset, subjects=subjects, return_epochs=True)

    #add domain_id == "session_subject_dataset" to metadata
    meta["domain_id"] = meta.session.astype(str) + "_" + meta.subject.astype(str) + "_" + dataset.code

    #reject bad epochs
    reject_criteria = dict(eeg=200e-6)  # threshold in Volts
    epochs.drop_bad(reject=reject_criteria, verbose="warning")

    #select the events corresponding to the remaining epochs
    valid_event_indices = epochs.selection  # This contains indices of epochs kept
    y = y[valid_event_indices]
    meta = meta.iloc[valid_event_indices]
    meta = meta.reset_index(drop=True)

    if classes_of_int:
        idx_clss = np.where(np.isin(y, classes_of_int))[0]
        epochs = epochs[idx_clss]
        y = y[idx_clss]
        meta = meta.iloc[idx_clss]


    #filter data
    if isinstance(paradigm, P300):
        l_freq = 1.0
        h_freq = 24.0

    elif isinstance(paradigm, MotorImagery) or isinstance(paradigm, LeftRightImagery):
        l_freq = 8.0
        h_freq = 32.0

    else:
        raise NotImplementedError(f"Paradigm {paradigm} not supported yet.")


    epochs = epochs.copy().filter(l_freq=l_freq, h_freq=h_freq,
                                    method='iir',
                                    iir_params=dict(order=4, ftype='butter'),
                                    phase='zero'
                                )


    #data as numpy array (epochs x channels x times)
    X = epochs.get_data()

    return X, y, meta

def adapt_single_trial(X_test_covs, X_train_covs, op="whitening"):
    """
    Transform each test covariance matrix using statistics from the train
    op: string,
        "whitening", for whiten by train mean
        "parallel", for parallel transport from the test sample to the train mean
    """
    spdm = SPDManifold()
    #train Fréchet mean (TODO: maybe should use the one learnt in train, per domain)
    mean = spdm.karcher_flow(X_train_covs, steps=2)

    if op=="whitening":
        P = spdm._matrix_invsqrt(mean)
        X_test_covs = P @ X_test_covs @ P

    if op=="parallel":
        # X_test_covs = spdm.parallel_transp_to_id
        pass
    return X_test_covs

def eeg_transform(X, 
                  paradigm,
                  y=None, 
                  xdawn_filters=4,
                  normalize=False, 
                  estimator = "lwf", 
                  epsilon=1.1618, 
                  transf = None):
        """
        Applies transformations to time series. 
        X is a (epochs, channels, times) time-series array and y is epochs-length array

        Applies transformation in this order:
        Xdawn filtering if xdawn_filters > 0
        Normalization of data if normalize = True
        Covariance estimation using pyrieamann supertrial 
            if covariance_estimation = "ERPCovariances" (using estimator argument; default to 'lwf')
            else use regular covariance estimation (and regularize using epsilon)

        Returns
            transformed tensor X 
            label array y 
        """

        #Xdawn
        if xdawn_filters > 0: 
            xd = Xdawn(nfilter=4, classes=["Target"])
            X = xd.fit_transform(X, y)

        #Use µV
        X = 1e6*X

        #normalization
        if normalize:
            #normalize each dim/channel using timepoints from all datasamples/epochs together (apparently MNE does that)
            X -= np.mean(X, axis=(0,2), keepdims=True)
            X /= np.std(X, axis=(0,2), keepdims=True)

        #covariance estimation
        if isinstance(paradigm, P300):
            print("Using ERPCovariances for P300 paradigm")
            if transf is not None: #for test and valid
                X = transf.transform(X)
            elif y is not None: #for train
                transf = ERPCovariances(classes=["Target"], estimator = estimator)
                X = transf.fit_transform(X, y)
            else: 
                raise ValueError("Either transformation or labels y should be provided")
            X += epsilon * np.eye(X.shape[-1])  #Tikhonov Regularization #TODO test without it


        elif isinstance(paradigm, MotorImagery) or isinstance(paradigm, LeftRightImagery):
            print("Using Covariances for Motor Imagery paradigm")
            X = Covariances(estimator=estimator).transform(X)
            # X += epsilon * np.eye(X.shape[-1])  #Tikhonov Regularization

        else:
            raise ValueError("Paradigm not supported. Use P300 or MotorImagery.")
        
        X = torch.Tensor(X)

        return X, transf


class EEGDataset(torch.utils.data.Dataset):
    """
        Dataset of covariance matrices of SyntheticSeriesDataset (one per epoch)
    """
    def __init__(self, data, labels, original_ts, metadata, transform=None):
        self.data = data
        self.str_labels = labels
        self.original_ts = original_ts
        self.metadata = metadata
        self.transform = transform

        self.classes = np.unique(self.str_labels)
        self.numerical_labels = torch.Tensor([np.where(self.classes==i)[0][0] for i in self.str_labels]).to(torch.int64)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        '''
        Generates one sample
        '''
        data, numerical_labels = self.data[idx], self.numerical_labels[idx]
        if self.transform:
            data = self.transform(data)
        return torch.Tensor(data), numerical_labels, self.metadata["domain_id"].iloc[idx]


class DomainBatchSampler(Sampler):
    def __init__(self, domain_ids, batch_size, shuffle=True, drop_last=False, min_batch_size=2):
        """
        Creates batches with unique domains (session or session/subject or session/subject/db) 
        with options to drop last batch if batch_size is not reached or more loosely if it doesn't reach min_batch_size
        (domains and samples within domains can be randomized while keeping the unicity of domains in each batch)
        
        domain_ids: list or array-like of domain id numbers (length = n_epochs)
        batch_size: number of samples per batch
        shuffle: whether to shuffle domains and samples within domains
        """
        self.domain_ids = np.array(domain_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.min_batch_size = min_batch_size

        #group sample indices by domain
        self.domain_to_indices = defaultdict(list)
        for idx, dom in enumerate(self.domain_ids):
            self.domain_to_indices[dom].append(idx)
        
        self.domains = list(self.domain_to_indices.keys())

    def __iter__(self):
        domains = self.domains.copy()
        if self.shuffle:
            np.random.shuffle(domains) #shuffle domain instead of all data samples

        all_batches = []
    
        for dom in domains:
            indices = self.domain_to_indices[dom]
            if self.shuffle:
                np.random.shuffle(indices)  #shuffle data samples inside each domain
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                if self.drop_last:    #drop last if length is inferior to batch_size
                    if len(batch) == self.batch_size:
                        all_batches.append(batch)
                else:
                    if len(batch) >= self.min_batch_size: #drop batch if length is inferior to min_batch_size
                        all_batches.append(batch)

        if self.shuffle:
            np.random.shuffle(all_batches) #shuffle batches order (domains/sessions might appear non-contiguously)

        for batch in all_batches:
            yield batch

    def __len__(self):
        total_batches = 0
        for indices in self.domain_to_indices.values():
            n = len(indices)
            n_batches = n // self.batch_size
            if not self.drop_last and n % self.batch_size >= self.min_batch_size:
                n_batches += 1
            total_batches += n_batches
        return total_batches

#FIXED: split 1st, ERP estimation after
def get_eeg_transform_all_domains(dataset, 
                                  paradigm,
                                  classes_of_int = None,
                                  subjects=None,
                                **eeg_transform_params):
    """
    Given a dataset and its paradigm, treats, creates train test split and estimates data descriptor (cov matrices)
    by considering each domain separetly (and label stratified). Then, concatenates the split from all domains
    Returns 
    (X_train, y_train, meta_train, time_series_train), (X_test, y_test, meta_test, time_series_test)
    ensuring 
        every domain is present in both train and test and 
        label balance between train and test for every domain
    """
    time_series, y, meta = moabb_to_epochs(dataset=dataset, paradigm=paradigm, classes_of_int=classes_of_int, subjects=subjects)
    domains = np.unique(meta["domain_id"])

    X_doms_train = []
    y_doms_train = []
    meta_doms_train = []
    ts_doms_train = [] 

    X_doms_val = []
    y_doms_val = []
    meta_doms_val = []
    ts_doms_val = []

    X_doms_test = []
    y_doms_test = []
    meta_doms_test = []
    ts_doms_test = []

    #debug
    X_doms_test_off = []

    test_perc = 0.15
    val_perc = 0.15

    
    #transform each domain separetly, split each domain separetly and then concatenate train (val, test) from all dom
    for dom in domains:
        meta_dom = meta[meta.domain_id == dom]
        y_dom = y[meta.domain_id == dom]
        ts_dom = time_series[meta.domain_id == dom]


        #split train and test
        y_dom_train, y_dom_test, \
            ts_dom_train, ts_dom_test, \
            meta_dom_train, meta_dom_test = train_test_split(y_dom, ts_dom, 
                                                            meta_dom, test_size=test_perc, 
                                                            random_state=42, stratify=y_dom)
        #take a percentage of train for validation 
        y_dom_train, y_dom_val, \
            ts_dom_train, ts_dom_val, \
            meta_dom_train, meta_dom_val = train_test_split(y_dom_train, ts_dom_train, 
                                                            meta_dom_train, test_size=val_perc/(1-test_perc), 
                                                            random_state=42, stratify=y_dom_train)
        
        #estimate covariance matrices within each split 
        X_dom_train, erp_transf = eeg_transform(X=ts_dom_train,
                                    y=y_dom_train, 
                                    paradigm=paradigm,
                                    **eeg_transform_params)
        X_dom_val, _ = eeg_transform(X=ts_dom_val,
                                  paradigm=paradigm,
                                  transf=erp_transf,
                                  **eeg_transform_params)
        X_dom_test, _ = eeg_transform(X=ts_dom_test,
                                   paradigm=paradigm,
                                   transf=erp_transf,
                                   **eeg_transform_params)
        #debug leakage
        # X_dom_test_off, _ = eeg_transform(X=ts_dom_test,
        #                             y=y_dom_test, 
        #                             paradigm=paradigm,
        #                             **eeg_transform_params)

        #debug: 
        X_dom_test_off, _ = eeg_transform(X=ts_dom_test,
                                   paradigm=paradigm,
                                   transf=erp_transf,
                                   **eeg_transform_params)
        #debug:whiten by domain train mean
        X_dom_test_off = adapt_single_trial(X_test_covs=X_dom_test_off, X_train_covs=X_dom_train)

        X_doms_test_off.append(X_dom_test_off)

        X_doms_train.append(X_dom_train)
        y_doms_train.extend(y_dom_train)
        meta_doms_train.append(meta_dom_train)
        ts_doms_train.append(ts_dom_train)

        X_doms_val.append(X_dom_val)
        y_doms_val.extend(y_dom_val)
        meta_doms_val.append(meta_dom_val)
        ts_doms_val.append(ts_dom_val)

        X_doms_test.append(X_dom_test)
        y_doms_test.extend(y_dom_test)
        meta_doms_test.append(meta_dom_test)
        ts_doms_test.append(ts_dom_test)


    X_train = torch.vstack(X_doms_train)
    y_train = np.array(y_doms_train)
    meta_train = pd.concat(meta_doms_train)
    ts_train = np.vstack(ts_doms_train)

    X_val = torch.vstack(X_doms_val)
    y_val = np.array(y_doms_val)
    meta_val = pd.concat(meta_doms_val)
    ts_val = np.vstack(ts_doms_val)

    X_test = torch.vstack(X_doms_test)
    y_test = np.array(y_doms_test)
    meta_test = pd.concat(meta_doms_test)
    ts_test = np.vstack(ts_doms_test)

    #debug
    X_test_off = torch.vstack(X_doms_test_off)

    return (X_train, y_train, meta_train, ts_train), \
           (X_val, y_val, meta_val, ts_val), \
           (X_test, y_test, meta_test, ts_test), \
           (X_test_off, y_test, meta_test, ts_test) #debug



def get_eeg_dataloader_treated_by_domain(moabb_dataset, paradigm, batch_size, min_batch_size, classes_of_int=None, subjects = None, **eeg_transform_params):
    
    (X_train, y_train, meta_train, ts_train), \
    (X_val, y_val, meta_val, ts_val), \
    (X_test, y_test, meta_test, ts_test), \
    (X_test_off, _, _, _)  = get_eeg_transform_all_domains(dataset=moabb_dataset, 
                                                           paradigm=paradigm, 
                                                           classes_of_int=classes_of_int,
                                                           subjects=subjects,
                                                           **eeg_transform_params)
    #extra debug arg above
    dataset_train = EEGDataset(data=X_train,labels=y_train,original_ts=ts_train, metadata=meta_train)
    dataset_val = EEGDataset(data=X_val,labels=y_val,original_ts=ts_val, metadata=meta_val)
    dataset_test = EEGDataset(data=X_test,labels=y_test,original_ts=ts_test, metadata=meta_test)
    #debug
    dataset_test_off = EEGDataset(data=X_test_off,labels=y_test,original_ts=ts_test, metadata=meta_test)

    #samplers for iterating over domains/sessions respecting min_batch_size
    sampler_train = DomainBatchSampler(
        meta_train['domain_id'],
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,       #keep small batches
        min_batch_size=min_batch_size       #avoid batches with less than min_batch_size
    )
    sampler_test = DomainBatchSampler(
        meta_test['domain_id'],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,       #keep small batches
        min_batch_size=min_batch_size       #avoid batches with less than min_batch_size
    )
    sampler_val = DomainBatchSampler(
        meta_val['domain_id'],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,       #keep small batches
        min_batch_size=min_batch_size       #avoid batches with less than min_batch_size
    )
    train_loader = DataLoader(dataset_train, batch_sampler = sampler_train)
    val_loader = DataLoader(dataset_val, batch_sampler=sampler_val)
    test_loader = DataLoader(dataset_test, batch_sampler=sampler_test)
    #debug
    test_loader_off = DataLoader(dataset_test_off, batch_sampler=sampler_test)
    return train_loader, val_loader, test_loader, test_loader_off



if __name__== "__main__":
    dataset = BI2013a()
    paradigm = P300()
    X, y, meta = moabb_to_epochs(dataset, paradigm)
