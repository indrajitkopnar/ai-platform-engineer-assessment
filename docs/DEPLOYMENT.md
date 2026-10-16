# Deployment

Three levels, from simplest to production-grade. All of them run **the same Docker image**.

---

## Option A — Docker on any machine (simplest)

```bash
cp .env.example .env          # optional: set LLM_PROVIDER etc.; without it the mock LLM is used
docker compose up --build -d
curl http://localhost:8000/health
docker compose logs -f api    # follow logs
docker compose down           # stop
```
What the image does ([`Dockerfile`](../Dockerfile)): installs pinned dependencies, **trains the classifier at build
time** (the image carries a fixed model version), runs as a **non-root user**, defines a **health check**, and
starts Uvicorn with 2 worker processes.

## Option B — A Linux server (VM) behind nginx with HTTPS

```
 internet / company network ──HTTPS 443──► nginx (TLS, rate limit, request size limit)
                                              └─► 127.0.0.1:8000  docker container (api)
```

**1. Prepare the server** (Ubuntu example)
```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 nginx git
sudo usermod -aG docker $USER      # log out and in again
git clone https://github.com/indrajitkopnar/ai-platform-engineer-assessment.git
cd ai-platform-engineer-assessment
cp .env.example .env && nano .env  # real LLM settings; keep this file readable only by the deploy user
chmod 600 .env
```

**2. Bind the API to localhost only** — change the port line in `docker-compose.yml` to `"127.0.0.1:8000:8000"`,
so only nginx can reach it. Then `docker compose up -d --build`.

**3. nginx site** (`/etc/nginx/sites-available/ai-assessment`):
```nginx
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

server {
    listen 443 ssl;
    server_name ai.example.internal;

    ssl_certificate     /etc/ssl/certs/ai.crt;       # company certificate or Let's Encrypt (certbot)
    ssl_certificate_key /etc/ssl/private/ai.key;

    client_max_body_size 1m;

    location / {
        limit_req zone=api burst=20 nodelay;
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Request-Id $request_id;     # same id appears in the API logs
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 60s;
    }
}
server { listen 80; server_name ai.example.internal; return 301 https://$host$request_uri; }
```
```bash
sudo ln -s /etc/nginx/sites-available/ai-assessment /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

**4. Updates and rollback**
```bash
git pull && docker compose up -d --build                   # deploy a new version
git checkout <previous-tag> && docker compose up -d --build # roll back
```
`restart: unless-stopped` in the compose file brings the container back after a crash or reboot.

**Authentication:** in production, put the API behind the company identity provider (for example OAuth2/OIDC at
nginx or an API gateway). The chatbot's `user_groups` must then come **from the user's token**, not from the
request body.

## Option C — Kubernetes (for scale and high availability)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: ai-assessment-api }
spec:
  replicas: 2
  selector: { matchLabels: { app: ai-assessment-api } }
  template:
    metadata: { labels: { app: ai-assessment-api } }
    spec:
      containers:
        - name: api
          image: registry.example.com/ai-assessment-api:1.0.0
          ports: [{ containerPort: 8000 }]
          envFrom:
            - configMapRef: { name: ai-assessment-config }   # non-secret settings
            - secretRef:    { name: ai-assessment-secrets }  # LLM_API_KEY (from a vault / sealed secret)
          resources:
            requests: { cpu: 250m, memory: 512Mi }
            limits:   { cpu: "1",  memory: 1Gi }
          readinessProbe: { httpGet: { path: /health, port: 8000 }, initialDelaySeconds: 5 }
          livenessProbe:  { httpGet: { path: /health, port: 8000 }, periodSeconds: 20 }
---
apiVersion: v1
kind: Service
metadata: { name: ai-assessment-api }
spec:
  selector: { app: ai-assessment-api }
  ports: [{ port: 80, targetPort: 8000 }]
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: { name: ai-assessment-api }
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: ai-assessment-api }
  minReplicas: 2
  maxReplicas: 10
  metrics: [{ type: Resource, resource: { name: cpu, target: { type: Utilization, averageUtilization: 70 } } }]
```
Plus an Ingress (TLS) in front. The service is **stateless** (model inside the image, index built at startup), so
scaling is just adding replicas. For a large document collection, move the index into pgvector/OpenSearch so pods
don't each rebuild it.

## CI/CD

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every push and pull request:
1. install pinned dependencies,
2. **train the classifier and print metrics**,
3. **run the 18 tests**,
4. **run the chatbot golden-set evaluation**,
5. build the Docker image.

Production extension: push the image to a registry tagged with the commit SHA, deploy to a staging environment,
run the evaluation against it, then promote to production with an approval step (or GitOps with Argo CD).

## Configuration and secrets

| Setting | Where it lives |
|---|---|
| Non-secret configuration (provider, model, thresholds) | `.env` on a server · ConfigMap on Kubernetes |
| `LLM_API_KEY` | `.env` with `chmod 600` on a server · Kubernetes Secret fed from a vault (Azure Key Vault / AWS Secrets Manager) · **never in Git** |

## Monitoring in production

| What | How |
|---|---|
| Request rate, errors, latency (p50/p95/p99) | Logs already include request id, path, status and duration; add Prometheus metrics or OpenTelemetry |
| Classifier quality | Share handled automatically, LLM share, human-review share, agent override rate, class mix drift |
| Chatbot quality | Answered vs refused rate, which guardrail fires, user feedback, golden-set score on each release |
| LLM cost | Tokens and cost per request and per feature (logged in the gateway) |
| Alerts | Error rate or latency above target; sudden rise in refusals or human-review share (signals drift or an outage) |
