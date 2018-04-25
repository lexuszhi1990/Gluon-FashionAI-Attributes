
    def predict(self, test_dataset_path, model_path, task, symbol=None, network='densenet201'):
    # def predict_(self, test_dataset_path, model_path, task, symbol=None, network='densenet201'):
        logging.info('starting predict for %s.' % task)

        if not self.output_submission_path.exists():
            self.output_submission_path.mkdir()
            logging.info('create %s' % self.output_submission_path)

        results_path = self.output_submission_path.joinpath('%s.csv'%(task))
        f_out = results_path.open('w+')

        ctx = self.get_ctx()[0]
        if symbol is None:
            net = get_pretrained_model(network, task_class_num_list[task], ctx)
            net.load_params(model_path, ctx=ctx)
            logging.info("load model from %s" % model_path)
        else:
            net = symbol

        # val_data = self.get_validate_data(task)
        test_data = self.get_gluon_dataset(test_dataset_path ,task, dataset_type='test')
        logging.info("load test dataset from %s" % (test_dataset_path))
        for i, batch in enumerate(test_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=[ctx], batch_axis=0, even_split=False)
            outputs = [net(X) for X in data]
            for output in outputs[0]:
                import pdb
                pdb.set_trace()
                out = nd.softmax(output)
                pred_out = ';'.join(["%.8f"%(o) for o in out.asnumpy().tolist()])
                line_out = ','.join([path, task, pred_out])
                f_out.write(line_out + '\n')
        f_out.close()
        logging.info("end predicting for %s, results saved at %s" % (task, results_path))
