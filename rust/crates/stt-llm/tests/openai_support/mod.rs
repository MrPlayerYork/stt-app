//! Tiny fake HTTP server on 127.0.0.1 (one request per connection) that records every request.
#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{Value, json};

#[derive(Debug, Clone)]
pub struct Req {
    pub method: String,
    pub path: String,
    pub auth: Option<String>,
    pub body: Option<Value>,
}

pub struct Reply {
    pub code: u16,
    pub ctype: &'static str,
    pub body: Vec<u8>,
}

impl Reply {
    pub fn json(code: u16, v: Value) -> Self {
        Self {
            code,
            ctype: "application/json",
            body: v.to_string().into_bytes(),
        }
    }
    pub fn text(code: u16, t: &str) -> Self {
        Self {
            code,
            ctype: "text/plain",
            body: t.as_bytes().to_vec(),
        }
    }
}

pub struct Fake {
    /// `http://127.0.0.1:<port>/v1`
    pub url: String,
    log: Arc<Mutex<Vec<Req>>>,
}

impl Fake {
    pub fn start(handler: impl Fn(&Req) -> Reply + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let log = Arc::new(Mutex::new(Vec::new()));
        let handler = Arc::new(handler);
        let log2 = log.clone();
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (h, l) = (handler.clone(), log2.clone());
                thread::spawn(move || serve(stream, &*h, &l));
            }
        });
        Self {
            url: format!("http://127.0.0.1:{port}/v1"),
            log,
        }
    }

    pub fn log(&self) -> Vec<Req> {
        self.log.lock().unwrap().clone()
    }

    /// Bodies POSTed to the chat completions endpoint.
    pub fn chat_bodies(&self) -> Vec<Value> {
        self.log()
            .into_iter()
            .filter(|r| r.method == "POST" && r.path == "/v1/chat/completions")
            .filter_map(|r| r.body)
            .collect()
    }
}

fn serve(mut s: TcpStream, handler: &dyn Fn(&Req) -> Reply, log: &Mutex<Vec<Req>>) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        match s.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let method = first.next().unwrap_or("").to_string();
    let path = first.next().unwrap_or("").to_string();
    let (mut len, mut auth) = (0usize, None);
    for l in lines {
        if let Some((k, v)) = l.split_once(':') {
            match k.trim().to_ascii_lowercase().as_str() {
                "content-length" => len = v.trim().parse().unwrap_or(0),
                "authorization" => auth = Some(v.trim().to_string()),
                _ => {}
            }
        }
    }
    while buf.len() < head_end + len {
        match s.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    }
    let req = Req {
        method,
        path,
        auth,
        body: serde_json::from_slice(&buf[head_end..head_end + len]).ok(),
    };
    log.lock().unwrap().push(req.clone());
    let r = handler(&req);
    let head = format!(
        "HTTP/1.1 {} X\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        r.code,
        r.ctype,
        r.body.len()
    );
    let _ = s.write_all(head.as_bytes());
    let _ = s.write_all(&r.body);
}

/// A chat completion response with `message` merged into the assistant message.
pub fn chat(message: Value, usage: Option<Value>) -> Value {
    let mut msg = json!({"role": "assistant"});
    for (k, v) in message.as_object().unwrap() {
        msg[k] = v.clone();
    }
    let mut d = json!({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": msg}]});
    if let Some(u) = usage {
        d["usage"] = u;
    }
    d
}
