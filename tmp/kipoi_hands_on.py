import kipoi


def handsOnDeepSeaBeluga():
    model = kipoi.get_model('DeepSEA/beluga')

    print(f"{model = }")

    pred = model.pipeline.predict_example(batch_size=4)

    # Download example dataloader kwargs
    dl_kwargs = model.default_dataloader.download_example('example')
    # Get the dataloader and instantiate it
    dl = model.default_dataloader(**dl_kwargs)
    # get a batch iterator
    batch_iterator = dl.batch_iter(batch_size=4)
    for batch in batch_iterator:
        # predict for a batch
        print(batch)
        batch_pred = model.predict_on_batch(batch['inputs'])

    pred = model.pipeline.predict(dl_kwargs, batch_size=4)
    pass

if __name__ == '__main__':
    handsOnDeepSeaBeluga()
    pass
