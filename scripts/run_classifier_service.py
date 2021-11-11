
import asyncio
import nest_asyncio # NB only for google cloud
nest_asyncio.apply()

from pubsub import KafkaServer, GoogleServer

from fmnist_service.model_runner import ModelRunner

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--gcloud', default=False)

    parser.add_argument('--project_id', default='vectorai-331519')
    parser.add_argument('--subscription_id', default='fmnist_listener') # 'fmnist_results_listener'

    parser.add_argument('--save_path', default='./fmnist_network.pth')
    parser.add_argument('--request_topic_id', default='fmnist_request')
    parser.add_argument('--result_topic_id', default='fmnist_result')

    parser.add_argument('--server_address', default='localhost:9092')
    parser.add_argument('--max_batch_size', type=int, default=100)

    return parser.parse_args()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    args = parse_args()

    if args.gcloud:
        server = GoogleServer(project_id=args.project_id, 
                              request_topic_id=args.request_topic_id,
                              result_topic_id=args.result_topic_id, 
                              subscription_id=args.subscription_id, loop=loop)
    else:
        server = KafkaServer(request_topic_id=args.request_topic_id,
                            result_topic_id=args.result_topic_id, 
                            server_address=args.server_address, loop=loop)

    runner = ModelRunner(args.save_path, batch_size=args.max_batch_size)
    task = loop.create_task(runner())

    server.start_transaction_service(runner.process_request)
    asyncio.wait_for(task) # idelaly should be awaited




