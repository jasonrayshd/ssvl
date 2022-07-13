# test the connectivity of network

import socket
import sys


def server():
    s =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    assert len(sys.argv) >= 3, f"expected 3 arguments, only got {len(sys.argv)}"
    s.bind(("0.0.0.0", int(sys.argv[2])))
    s.listen()
    conn, addr = s.accept()
    print(addr)
    conn.sendall(b"sucess")
    conn.close()
    

def client():
    s =  socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    assert len(sys.argv) >= 4, f"expected 4 arguments, only got {len(sys.argv)}"
    s.connect((sys.argv[3], int(sys.argv[2])))
    msg = s.recv(1024)
    print(msg.decode("utf-8"))


if __name__ == "__main__":
    com = sys.argv[1]

    if com == "server":
        server()
    else:
        client()