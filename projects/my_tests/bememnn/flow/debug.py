from parlai.scripts.train_model import TrainModel


if __name__ == '__main__':

    # TrainModel.main(
    #     model='bememnn',
    #     # model='bert_ranker/bi_encoder_ranker',
    #     # model='memnn',
    #     batchsize=3,
    #     task='flow:flow:3',
    #     model_file='bememnn_test/bememnn',
    # )

    TrainModel.main(
        init_opt='settings_1.json',
        batchsize=1,
    )