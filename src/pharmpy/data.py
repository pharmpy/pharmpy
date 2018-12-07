import collections
import numpy as np
import pandas as pd
from pathlib import Path

import pharmpy.math

#TODO: We could potentially have a mixin for filesaves of iterated dataframes
class DatasetIterator:
    """ Base class for iterator classes that generate datasets
    """
    def __iter__(self):
        return self

    def data_files(self, path, filename="resample_{}.dta"):
        """
            path is the path to where to create the resampled datasets
            filename is a filename format where the number of the resample will be inserted starting from 1
        """
        path = Path(path) 
        self.paths = []
        for df, i in enumerate(self, 1):
            next_path = path / filename.format(i)
            df.to_csv(next_path)
            self.paths.append(next_path)

    # Attribute paths can be used to get the paths to files stored. Document this


class GroupOmitter(DatasetIterator):
    """Iterate over omissions of a certain group in a dataset. One group omitted at a time
    """
    def __init__(self, df, group):
        self._unique_groups = df[group].unique()
        self._df = df
        self._counter = 0

    def __next__(self):
        if self._counter == len(self._unique_groups):
            raise StopIteration
        df = self._df
        next_group = self._unique_groups[self._counter]
        new_df = df[df[group] != next_group]
        self._counter = self._counter + 1
        return new_df


class Resampler(DatasetIterator):
    # TODO: Proper documentation
    """Iterate over resamples of a dataset.

       The dataset will be grouped on the group column
       then groups will be selected randomly with or without replacement to
       form a new dataset. Stratification will make sure that 

       group is the name of the group column.
       the groups will be renumbered from 1 and upwards
       stratify is the name of the stratification column
            the values in the stratification column must be equal within a group so that the group
            can be uniquely determined. The method will raise an exception otherwise.
       sample_size is the number of groups that should be sampled. The default is the number of groups
            If using stratification the default is to sample using the proportion of the stratas in the dataset
            A dictionary of specific sample sizes for each strata can also be supplied.
       replace is a boolean controlling whether sampling should be done with or without replacement

       Returns a tuple of a resampled DataFrame and a list of resampled groups in order
    """

    def __init__(self, df, group, resamples=1, stratify=None, sample_size=None, replace=False):
        unique_groups = df[group].unique()
        numgroups = len(unique_groups)

        if sample_size is None:
            sample_size = numgroups

        if stratify:
            # Default is to use proportions in dataset
            stratas = df.groupby(stratify)[group].unique()
            have_mult_sample_sizes = isinstance(sample_size, collections.Mapping)
            if not have_mult_sample_sizes:
                non_rounded_sample_sizes = stratas.apply(lambda x: (len(x) / numgroups) * sample_size)
                rounded_sample_sizes = pharmpy.math.round_and_keep_sum(non_rounded_sample_sizes, sample_size)
                sample_size_dict = dict(rounded_sample_sizes)    # strata: numsamples
            stratas = dict(stratas)     # strata: list of groups
        else:
            sample_size_dict = {1: sample_size}
            stratas = {1: unique_groups} 

        # Check that we will not run out of samples without replacement.
        if not replace:
            for strata in sample_size_dict:
                if sample_size_dict[strata] > len(stratas[strata]):
                    if stratify:
                        raise ValueError('The sample size ({sample_size}) for strata {strata} is larger than the number of groups' \
                            ' ({numgroups}) in that strata which is impoosible with replacement.'.format(
                                sample_size=sample_size_dict[strata], strata=strata, numgroups=len(stratas[strata])))
                    else:
                        raise ValueError('The sample size ({sample_size}) is larger than the number of groups' \
                            '({numgroups}) which is impossible with replacement.'.format(sample_size=sample_size_dict[strata], numgroups=len(stratas[strata]))) 

        self._df = df
        self._group = group
        self._replace = replace
        self._stratas = stratas
        self._sample_size_dict = sample_size_dict
        self._resamples = resamples

    def __next__(self):
        """
            generate resampled DataFrames
        """
        for i in range(0, self._resamples):
            random_groups = []
            for strata in self._sample_size_dict:
                random_groups += list(np.random.choice(self._stratas[strata], size=self._sample_size_dict[strata], replace=self._replace))

            new_df = pd.DataFrame()
            # Build the dataset given the random_groups list
            for grp_id, new_grp in zip(random_groups, range(1, len(random_groups) + 1)):
                sub = self._df.loc[self._df[self._group] == grp_id].copy()
                sub[self._group] = new_grp
                new_df = new_df.append(sub)

            return (new_df, list(random_groups))
