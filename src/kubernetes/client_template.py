client_template = """
apiVersion: batch/v1
kind: Job
metadata:
  name: vllm-client-{{ cfg.name }}
  namespace: {{ cfg.namespace }}
spec:
  template:
    spec:
      containers:
        - name: vllm-client
          image: {{ cfg.client_image }}
          command: ["python3", "/app/vllm_prompting_job.py",
                    "--server_name=vllm-server-{{ cfg.name }}",
                    "--namespace={{ cfg.namespace }}",
                    "--load_from_checkpoint={{ cfg.load_from_checkpoint }}",
                    "--budget={{ cfg.Client_Job.budget }}",
                    "--num_choices={{ cfg.Client_Job.num_choices }}",
                    "--num_samples={{ cfg.Client_Job.num_samples }}",
                    "--concurrency={{ cfg.Client_Job.concurrency }}",
                    "--batch_size={{ cfg.Client_Job.batch_size }}",
                    "--start_verifier={{ cfg.Client_Job.start_verifier }}",
                    "--temperatures={{ cfg.Client_Job.temperatures }}",
                    "--max_tokens={{ cfg.Client_Job.max_tokens }}",
                    "--thresholds={{ cfg.Client_Job.thresholds }}",
                    "--eval_mode={{ cfg.Client_Job.eval_mode }}",
                    "--ood_eval={{ cfg.Client_Job.ood_eval }}",
                    "--merlin_mode={{ cfg.Client_Job.merlin_mode }}",
                    "--guided_decoding={{ cfg.Client_Job.guided_decoding }}",
                    "--guided_reasoning={{ cfg.Client_Job.guided_reasoning }}",
                    "--think_about_labs={{ cfg.Client_Job.think_about_labs }}",
                    "--lora={{ cfg.Model.lora }}",
                    "--lora_name={{ cfg.Model.lora_modules }}",
                    "--store_patients={{ cfg.Client_Job.store_patients }}",
                    "--hardware_string={{ cfg.hardware_string }}",
                    "--config_string={{ cfg }}",
                    ]
          env:
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: wandb-secret
                  key: api-key
          resources:
            limits:
              memory: "32Gi"
              # cpu: "4"
              nvidia.com/gpu: "1"
            requests:
              # memory: "4Gi"
              # cpu: "1"
              nvidia.com/gpu: "1"
        {% if cfg.namespace != 'clinibench' %}
          volumeMounts:
            - name: checkpoints-volume
              mountPath: /checkpoints
      volumes:
        - name: checkpoints-volume
          persistentVolumeClaim:
            claimName: checkpoints-volume
        {% endif %}
      imagePullSecrets:
        - name: private-registry-auth
      nodeSelector:
        gpu: {{ cfg.Hardware.client_gpu }}
        {% if cfg.Hardware.client_gpu == 'v100' %}
        # kubernetes.io/hostname: cl-worker22
        {% endif %}
      restartPolicy: Never
  backoffLimit: 0
"""