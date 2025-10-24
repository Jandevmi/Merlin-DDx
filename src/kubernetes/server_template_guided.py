server_template = """
apiVersion: v1
kind: Pod
metadata:
  name: vllm-server-{{ cfg.name }}
  namespace: {{ cfg.namespace }}
  labels:
    app: vllm-server-{{ cfg.name }}
spec:
  restartPolicy: Never
  containers:
    - name: vllm-server
      image: {{ cfg.server_image }}
      command: 
      - "/bin/bash"
      - "-c"
      - |
        pip install --no-cache-dir outlines && \
        python3 -m vllm.entrypoints.api_server \
          --model={{ cfg.Model.base_model }} \
          --max-model-len={{ cfg.Model.max_model_len }} \
          --max-num-seqs={{ cfg.Model.max_num_seqs }} \
          --max-num-batched-tokens={{ cfg.Model.max_num_batched_tokens }} \
          --tensor-parallel-size={{ cfg.Hardware.parallel_size }} \
          --download-dir=/models \
          {% if cfg.Model.lora %}
          --enable-lora \
          --lora-modules=medreason=/ft_models/merlin-ddx/output/{{ cfg.Model.lora_modules }} \
          --max-lora-rank={{ cfg.Model.max_lora_rank }} \
          {% endif %}
          {% if cfg.Model.thinking and 'v0.9.1' in cfg.server_image %}
          --enable-reasoning \
          {% endif %}
          {% if cfg.Model.thinking %}
          --reasoning-parser=deepseek_r1 \
          {% endif %}
          {% if cfg.Client_Job.guided_decoding %}
          --guided-decoding-backend=outlines \
          {% endif %}
      ports:
        - containerPort: 8000
      resources:
        limits:
          memory: "{{ cfg.memory_limit_server }}Gi"
          nvidia.com/gpu: "{{ cfg.Hardware.parallel_size }}"
        requests:
          memory: "{{ cfg.memory_request_server }}Gi"
          nvidia.com/gpu: "{{ cfg.Hardware.parallel_size }}"
      volumeMounts:
        - name: dshm
          mountPath: /dev/shm  # Mount shared memory
        - name: model-volume
          mountPath: /models
        {% if cfg.namespace == 'clinibench' %}
        - name: merlin-ddx-pvc
          mountPath: /ft_models
        {% endif %}
      env:
        - name: HUGGING_FACE_HUB_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: HF_TOKEN
        - name: NCCL_DEBUG
          value: "INFO"
        - name: GLOO_SOCKET_IFNAME
          value: "eth0"
  volumes:
    - name: dshm  # Define shared memory volume
      emptyDir:
        medium: Memory
    - name: model-volume
      persistentVolumeClaim:
        claimName: model-volume
    {% if cfg.namespace == 'clinibench' %}
    - name: merlin-ddx-pvc
      persistentVolumeClaim:
        claimName: merlin-ddx-ckpts-pvc
    {% endif %}
  nodeSelector:
    gpu: {{ cfg.Hardware.server_gpu }}
    # kubernetes.io/hostname: cl-worker28
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-server-{{ cfg.name }}
  namespace: {{ cfg.namespace }}
spec:
  selector:
    app: vllm-server-{{ cfg.name }}
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
"""
