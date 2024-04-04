import pika
import subprocess
import os
import json


def run_gen_data(dataset_name):
    try:

        subprocess.run("cd Generate_XA_Data && python GenData.py --dataset {}".format(dataset_name), shell=True,
                       check=True)

        print(f"Generated data for dataset {dataset_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

# # 调用函数并传入数据集名称
# run_gen_ground_truth("my_dataset")


def callback(ch, method, properties, body):
    message_info = json.loads(body.decode())
    command = message_info.get('command')
    dataset_name = message_info.get('datasetName')
    numPerturbSamples = message_info.get('numPerturbSamples')
    topNode = message_info.get('topNode')

    # 执行GenData操作，返回结果
    result = run_gen_data(dataset_name)

    # 发送结果到消息队列
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='algorithm_result_queue')
    channel.basic_publish(exchange='', routing_key='algorithm_result_queue', body=result)
    connection.close()

def consume_messages():
    connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
    channel = connection.channel()
    channel.queue_declare(queue='algorithm_queue')
    channel.basic_consume(queue='algorithm_queue', on_message_callback=callback, auto_ack=True)
    channel.start_consuming()

if __name__ == '__main__':
    # consume_messages()

    # #1 GenData
    dataset_name = 'syn1'
    num_perturb_samples = '5'
    top_node = '5'
    #
    # print(f"Generating data for dataset {dataset_name}")
    # subprocess.run("cd Generate_XA_Data && python GenData.py --dataset {}".format(dataset_name), shell=True, check=True)
    #
    # #2 GenGroundTruth
    # print(f"Generating groundtruth for dataset {dataset_name}")
    # subprocess.run("cd Generate_XA_Data && python GenGroundTruth.py --dataset {}".format(dataset_name), shell=True, check=True)

    #3 Train GNN
    print(f"Generating GNN for dataset {dataset_name}")
    subprocess.run("cd Train_GNN_model && python train.py --dataset {}".format(dataset_name), shell=True, check=True)

    # #4 Explain GNN precision
    # print(f"Explaining for dataset {dataset_name}, number of perturbing is {num_perturb_samples}, top-k when k is {top_node}")
    # subprocess.run("cd Explain_GNN && python main_server.py --dataset {} --num_perturb_samples {} --topnode {}".format(dataset_name, num_perturb_samples, top_node), shell=True, check=True)


