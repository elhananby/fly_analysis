import multiprocessing as mp
import socket
import zmq
import time
import json
import numpy as np
from dataclasses import dataclass
from statistics import mean, stdev
import os

@dataclass
class TestMessage:
    x: float
    y: float
    z: float
    timestamp: float = 0.0
    
    def to_json(self):
        return json.dumps(self.__dict__)
    
    @staticmethod
    def from_json(json_str):
        data = json.loads(json_str)
        return TestMessage(**data)

def unix_domain_socket_server(socket_path, queue, n_messages):
    """Traditional IPC using Unix Domain Socket"""
    if os.path.exists(socket_path):
        os.unlink(socket_path)
        
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)
    
    conn, _ = server.accept()
    latencies = []
    
    try:
        for _ in range(n_messages):
            data = conn.recv(1024)
            receive_time = time.time_ns()
            msg = TestMessage.from_json(data.decode())
            latency = (receive_time - msg.timestamp) / 1_000_000  # to milliseconds
            latencies.append(latency)
            conn.sendall(b'ack')
    finally:
        conn.close()
        server.close()
        os.unlink(socket_path)
        queue.put(latencies)

def unix_domain_socket_client(socket_path, n_messages):
    """Client for Unix Domain Socket"""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(socket_path)
    
    try:
        for _ in range(n_messages):
            msg = TestMessage(x=1.0, y=2.0, z=3.0, timestamp=time.time_ns())
            client.sendall(msg.to_json().encode())
            client.recv(3)  # receive ack
    finally:
        client.close()

def zmq_ipc_publisher(ipc_path, n_messages):
    """ZMQ Publisher using IPC"""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"ipc://{ipc_path}")
    
    # Small delay to allow subscriber to connect
    time.sleep(0.1)
    
    try:
        for _ in range(n_messages):
            msg = TestMessage(x=1.0, y=2.0, z=3.0, timestamp=time.time_ns())
            socket.send_json(msg.__dict__)
    finally:
        socket.close()
        context.term()

def zmq_ipc_subscriber(ipc_path, queue, n_messages):
    """ZMQ Subscriber using IPC"""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"ipc://{ipc_path}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    latencies = []
    try:
        for _ in range(n_messages):
            msg = socket.recv_json()
            receive_time = time.time_ns()
            latency = (receive_time - msg['timestamp']) / 1_000_000
            latencies.append(latency)
    finally:
        socket.close()
        context.term()
        queue.put(latencies)

def zmq_tcp_publisher(port, n_messages):
    """ZMQ Publisher using TCP"""
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    
    # Small delay to allow subscriber to connect
    time.sleep(0.1)
    
    try:
        for _ in range(n_messages):
            msg = TestMessage(x=1.0, y=2.0, z=3.0, timestamp=time.time_ns())
            socket.send_json(msg.__dict__)
    finally:
        socket.close()
        context.term()

def zmq_tcp_subscriber(port, queue, n_messages):
    """ZMQ Subscriber using TCP"""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://localhost:{port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    latencies = []
    try:
        for _ in range(n_messages):
            msg = socket.recv_json()
            receive_time = time.time_ns()
            latency = (receive_time - msg['timestamp']) / 1_000_000
            latencies.append(latency)
    finally:
        socket.close()
        context.term()
        queue.put(latencies)

def run_unix_socket_benchmark(n_messages):
    """Run Unix Domain Socket benchmark"""
    socket_path = "/tmp/benchmark_socket"
    queue = mp.Queue()
    
    server_process = mp.Process(target=unix_domain_socket_server, 
                              args=(socket_path, queue, n_messages))
    client_process = mp.Process(target=unix_domain_socket_client, 
                              args=(socket_path, n_messages))
    
    server_process.start()
    time.sleep(0.1)  # Allow server to start
    client_process.start()
    
    client_process.join()
    server_process.join()
    
    return queue.get()

def run_zmq_ipc_benchmark(n_messages):
    """Run ZMQ IPC benchmark"""
    ipc_path = "/tmp/benchmark_zmq"
    queue = mp.Queue()
    
    subscriber_process = mp.Process(target=zmq_ipc_subscriber, 
                                  args=(ipc_path, queue, n_messages))
    publisher_process = mp.Process(target=zmq_ipc_publisher, 
                                 args=(ipc_path, n_messages))
    
    subscriber_process.start()
    time.sleep(0.1)  # Allow subscriber to connect
    publisher_process.start()
    
    publisher_process.join()
    subscriber_process.join()
    
    return queue.get()

def run_zmq_tcp_benchmark(n_messages):
    """Run ZMQ TCP benchmark"""
    port = 5555
    queue = mp.Queue()
    
    subscriber_process = mp.Process(target=zmq_tcp_subscriber, 
                                  args=(port, queue, n_messages))
    publisher_process = mp.Process(target=zmq_tcp_publisher, 
                                 args=(port, n_messages))
    
    subscriber_process.start()
    time.sleep(0.1)  # Allow subscriber to connect
    publisher_process.start()
    
    publisher_process.join()
    subscriber_process.join()
    
    return queue.get()

def print_stats(name, latencies):
    """Print statistics for the benchmark"""
    print(f"\n{name} Results (milliseconds):")
    print(f"Average latency: {mean(latencies):.3f}")
    print(f"Std deviation:   {stdev(latencies):.3f}")
    print(f"Minimum latency: {min(latencies):.3f}")
    print(f"Maximum latency: {max(latencies):.3f}")
    print(f"99th percentile: {np.percentile(latencies, 99):.3f}")

def main():
    n_messages = 100
    print(f"Running benchmarks with {n_messages} messages each...")
    
    # Run Unix Domain Socket benchmark
    unix_socket_latencies = run_unix_socket_benchmark(n_messages)
    print_stats("Unix Domain Socket", unix_socket_latencies)
    
    # Run ZMQ IPC benchmark
    zmq_ipc_latencies = run_zmq_ipc_benchmark(n_messages)
    print_stats("ZMQ IPC", zmq_ipc_latencies)
    
    # Run ZMQ TCP benchmark
    zmq_tcp_latencies = run_zmq_tcp_benchmark(n_messages)
    print_stats("ZMQ TCP", zmq_tcp_latencies)

if __name__ == "__main__":
    main()