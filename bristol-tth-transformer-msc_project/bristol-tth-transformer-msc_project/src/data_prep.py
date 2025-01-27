# Make output directory #
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"./model_training/AOTransformer_{current_time}"
os.makedirs(outdir, exist_ok=False)

## Specify dataset files to run over ##
path = "/cephfs/dice/projects/CMS/Hinv/ml_datasets_ul/UL{year}_ml_inputs/{dataset}.parquet"

datasets = [
    'ttH_HToInvisible_M125',
    'TTToSemiLeptonic',
]
years = ['2018']

files = [
    path.format(year=year, dataset=dataset)
    for dataset in datasets
    for year in years
]

## Data preprocessing ##
df = load_from_parquet(files)
df = remove_negative_events(df)
df["target"] = create_target_labels(df["dataset"])
apply_reweighting_per_class(df)
reweighting = torch.Tensor(df['weight_training'].values)
df["target"] = create_target_labels(df["dataset"])

X, y, pad_mask = awkward_to_inputs_parallel(df, n_processes=8, target_length=10)

# Now save X,y, pad_mask