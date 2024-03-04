import pika
import subprocess


def run_gen_ground_truth(dataset_name):
    try:
        # 切换到Generate_XA_Data目录
        subprocess.run("cd Generate_XA_Data", shell=True, check=True)

        # 执行python命令生成数据集
        command = f"python GenData.py --dataset {dataset_name}"
        subprocess.run(command, shell=True, check=True)

        print(f"Generated ground truth for dataset {dataset_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

# 调用函数并传入数据集名称
run_gen_ground_truth("my_dataset")


def callback(ch, method, properties, body):
    dataset_name = body.decode()
    # 执行GenGroundTruth操作，返回结果
    result = run_gen_ground_truth(dataset_name)

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
    consume_messages()
