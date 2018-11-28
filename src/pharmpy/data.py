import collections
import numpy as np
import pandas as pd


def resample(df, group, stratify=None, sample_size=None, replace=False):
    """Resample a dataset.

       The dataset will be grouped on the group column
       then groups will be selected randomly with or without replacement to
       form a new dataset. Stratification will make sure that 

       group is the name of the group column.
       stratify is the name of the stratification column
            the values in the stratification column must be equal within a group so that the group
            can be uniquely determined. The method will raise an exception otherwise.
       sample_size is the number of groups that should be sampled. The default is the number of groups
            If using stratification the default is to sample using the proportion of the stratas in the dataset
            A dictionary of specific sample sizes for each strata can also be supplied.
       replace is a boolean controlling whether sampling should be done with or without replacement

       Returns a tuple of a resampled DataFrame and a list of resampled groups in order
    """
    unique_groups = df[group].unique()
    numgroups = len(unique_groups)

    if sample_size is None:
        sample_size = numgroups

    if replace:
        if isinstance(sample_size, collections.Mapping):
            pass
        else:
            # sample_size is scalar
            if sample_size > numgroups:
                raise ValueError('The sample size ({sample_size}) is larger than the number of groups ({numgroups}) which is impossible with replacement.'.format(sample_size=sample_size, numgroups=numgroups)) 

    random_groups = np.random.choice(unique_groups, size=sample_size, replace=replace)

    new_df = pd.DataFrame()
    for grp_id, new_grp in zip(random_groups, range(1, len(random_groups) + 1)):
        sub = df.loc[df[group] == grp_id].copy()
        sub[group] = new_grp
        new_df = new_df.append(sub)

    return (new_df, list(random_groups))
    #TODO set limit of sample_size with replacement
