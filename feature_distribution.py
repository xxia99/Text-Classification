# Assign the columns into three different types: binary, numerical, and categorical; save the results in three lists.
# "target_label" should belong to the binary list from a scientic view,
# but it isn't used as a feature in the training progress.


binary = ['post-menopause',
          'human epidermal growth factor receptor 2 is positive',
          'estrogen receptor positive',
          'progesterone receptor positive',
          'prior hormonal therapy',
          'prior chemotherapy',
          'biopsy type',
          'sentinel node biospy',
          'axillary dissection',
          'target_label']
numerical = ['number of positive axillary nodes', 'tumor size']
categorical = ['race',
               'treatment',
               'tumor laterality',
               'cancer histologic grade']