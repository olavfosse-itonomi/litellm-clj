(ns litellm.providers.azure-openai
  "Azure OpenAI provider implementation for LiteLLM

   Azure OpenAI uses a different endpoint structure than standard OpenAI:
   - Endpoint: https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions?api-version={version}
   - Auth: api-key header instead of Bearer token

   Configuration:
   - :api-key - Azure OpenAI API key
   - :api-base - Azure OpenAI resource endpoint (e.g., https://myresource.openai.azure.com)
   - :api-version - API version (e.g., 2024-08-01-preview)
   - :deployment - Deployment name (can also be passed as the model)"
  (:require [litellm.streaming :as streaming]
            [litellm.errors :as errors]
            [hato.client :as http]
            [cheshire.core :as json]
            [clojure.tools.logging :as log]
            [clojure.string :as str]
            [clojure.core.async :as async :refer [go >!]]))

;; ============================================================================
;; Constants
;; ============================================================================

(def default-api-version "2024-08-01-preview")

;; ============================================================================
;; Message Transformations (OpenAI-compatible)
;; ============================================================================

(defn transform-messages
  "Transform messages to Azure OpenAI format (same as OpenAI)"
  [messages]
  (map (fn [msg]
         (let [base {:role (name (:role msg))
                    :content (:content msg)}]
           (cond-> base
             (:name msg) (assoc :name (:name msg))
             (:tool-call-id msg) (assoc :tool_call_id (:tool-call-id msg)))))
       messages))

(defn transform-tools
  "Transform tools to Azure OpenAI format"
  [tools]
  (when tools
    (map (fn [tool]
           (let [func (:function tool)]
             {:type (:tool-type tool "function")
              :function {:name (or (:function-name func) (:name func))
                        :description (or (:function-description func) (:description func))
                        :parameters (or (:function-parameters func) (:parameters func))}}))
         tools)))

(defn transform-tool-choice
  "Transform tool choice to Azure OpenAI format"
  [tool-choice]
  (cond
    (keyword? tool-choice) (name tool-choice)
    (map? tool-choice) tool-choice
    :else tool-choice))

;; ============================================================================
;; Response Transformations
;; ============================================================================

(defn transform-tool-calls
  "Transform Azure OpenAI tool calls to standard format"
  [tool-calls]
  (when tool-calls
    (map (fn [tool-call]
           {:id (:id tool-call)
            :type (:type tool-call)
            :function {:name (get-in tool-call [:function :name])
                      :arguments (get-in tool-call [:function :arguments])}})
         tool-calls)))

(defn transform-message
  "Transform Azure OpenAI message to standard format"
  [message]
  (cond-> {:role (keyword (:role message))
           :content (:content message)}
    (:tool_calls message) (assoc :tool-calls (transform-tool-calls (:tool_calls message)))
    (:function_call message) (assoc :function-call (:function_call message))
    (:reasoning_content message) (assoc :reasoning-content (:reasoning_content message))))

(defn transform-choice
  "Transform Azure OpenAI choice to standard format"
  [choice]
  {:index (:index choice)
   :message (transform-message (:message choice))
   :finish-reason (keyword (:finish_reason choice))})

(defn transform-usage
  "Transform Azure OpenAI usage to standard format"
  [usage]
  (when usage
    {:prompt-tokens (:prompt_tokens usage)
     :completion-tokens (:completion_tokens usage)
     :total-tokens (:total_tokens usage)}))

;; ============================================================================
;; URL Building
;; ============================================================================

(defn build-completion-url
  "Build the Azure OpenAI completion URL.

   Format: {api-base}/openai/deployments/{deployment}/chat/completions?api-version={version}"
  [config model]
  (let [api-base (str/replace (or (:api-base config) "") #"/$" "")
        deployment (or (:deployment config) model)
        api-version (or (:api-version config) default-api-version)]
    (str api-base "/openai/deployments/" deployment "/chat/completions?api-version=" api-version)))

(defn build-embedding-url
  "Build the Azure OpenAI embedding URL.

   Format: {api-base}/openai/deployments/{deployment}/embeddings?api-version={version}"
  [config model]
  (let [api-base (str/replace (or (:api-base config) "") #"/$" "")
        deployment (or (:deployment config) model)
        api-version (or (:api-version config) default-api-version)]
    (str api-base "/openai/deployments/" deployment "/embeddings?api-version=" api-version)))

;; ============================================================================
;; Error Handling
;; ============================================================================

(defn handle-error-response
  "Handle Azure OpenAI API error responses"
  [provider response]
  (let [status (:status response)
        body (:body response)
        error-info (get body :error {})
        message (or (:message error-info) "Unknown error")
        provider-code (:code error-info)
        request-id (get-in response [:headers "x-ms-request-id"])]
    (throw (errors/http-status->error
             status
             "azure-openai"
             message
             :provider-code provider-code
             :request-id request-id
             :body body))))

;; ============================================================================
;; Model and Cost Configuration
;; ============================================================================

(def default-cost-map
  "Default cost per token for Azure OpenAI models (in USD)
   Note: Azure pricing may differ from OpenAI direct pricing"
  {"gpt-4" {:input 0.00003 :output 0.00006}
   "gpt-4-turbo" {:input 0.00001 :output 0.00003}
   "gpt-4o" {:input 0.000005 :output 0.000015}
   "gpt-4o-mini" {:input 0.00000015 :output 0.0000006}
   "gpt-35-turbo" {:input 0.0000005 :output 0.0000015}
   "gpt-35-turbo-16k" {:input 0.000003 :output 0.000004}})

;; ============================================================================
;; Azure OpenAI Provider Implementation Functions
;; ============================================================================

(defn transform-request-impl
  "Azure OpenAI-specific transform-request implementation"
  [provider-name request config]
  (let [;; Note: model in the request body is not used by Azure, deployment is in URL
        ;; But we include it for consistency
        model (:model request)
        base {:model model
              :messages (transform-messages (:messages request))}]

    (cond-> base
      (:max-tokens request) (assoc :max_tokens (:max-tokens request))
      (:temperature request) (assoc :temperature (:temperature request))
      (:top-p request) (assoc :top_p (:top-p request))
      (:frequency-penalty request) (assoc :frequency_penalty (:frequency-penalty request))
      (:presence-penalty request) (assoc :presence_penalty (:presence-penalty request))
      (:stop request) (assoc :stop (:stop request))
      (contains? request :stream) (assoc :stream (:stream request))
      (:tools request) (assoc :tools (transform-tools (:tools request)))
      (:tool-choice request) (assoc :tool_choice (transform-tool-choice (:tool-choice request)))
      (:reasoning-effort request) (assoc :reasoning_effort (name (:reasoning-effort request))))))

(defn make-request-impl
  "Azure OpenAI-specific make-request implementation"
  [provider-name transformed-request thread-pool telemetry config]
  (let [url (build-completion-url config (:model transformed-request))]
    (errors/wrap-http-errors
      "azure-openai"
      #(let [start-time (System/currentTimeMillis)
             response (http/post url
                                 (conj {:headers {"api-key" (:api-key config)
                                                  "Content-Type" "application/json"
                                                  "User-Agent" "litellm-clj/1.0.0"}
                                        :body (json/encode transformed-request)
                                        :timeout (:timeout config 30000)
                                        :async? true
                                        :as :json}
                                       (when thread-pool
                                         {:executor thread-pool})))
             duration (- (System/currentTimeMillis) start-time)]

         ;; Handle errors if response has error status
         (when (>= (:status @response) 400)
           (handle-error-response :azure-openai @response))

         response))))

(defn transform-response-impl
  "Azure OpenAI-specific transform-response implementation"
  [provider-name response]
  (let [body (:body response)]
    {:id (:id body)
     :object (:object body)
     :created (:created body)
     :model (:model body)
     :choices (map transform-choice (:choices body))
     :usage (transform-usage (:usage body))}))

(defn supports-streaming-impl
  "Azure OpenAI-specific supports-streaming? implementation"
  [provider-name]
  true)

(defn supports-function-calling-impl
  "Azure OpenAI-specific supports-function-calling? implementation"
  [provider-name]
  true)

(defn get-rate-limits-impl
  "Azure OpenAI-specific get-rate-limits implementation"
  [provider-name]
  ;; Azure has different rate limits per deployment
  {:requests-per-minute 1000
   :tokens-per-minute 100000})

(defn health-check-impl
  "Azure OpenAI-specific health-check implementation"
  [provider-name thread-pool config]
  ;; Azure doesn't have a /models endpoint like OpenAI
  ;; We'll do a simple check that the API base is reachable
  (try
    (let [url (str (str/replace (or (:api-base config) "") #"/$" "")
                   "/openai/models?api-version="
                   (or (:api-version config) default-api-version))
          response (http/get url
                             (conj {:headers {"api-key" (:api-key config)}
                                    :timeout 5000}
                                   (when thread-pool
                                     {:executor thread-pool})))]
      (= 200 (:status response)))
    (catch Exception e
      (log/warn "Azure OpenAI health check failed" {:error (.getMessage e)})
      false)))

(defn get-cost-per-token-impl
  "Azure OpenAI-specific get-cost-per-token implementation"
  [provider-name model]
  (get default-cost-map model {:input 0.0 :output 0.0}))

;; ============================================================================
;; Streaming Support
;; ============================================================================

(defn transform-streaming-chunk-impl
  "Azure OpenAI-specific transform-streaming-chunk implementation"
  [provider-name chunk]
  (let [choice (first (:choices chunk))
        delta (:delta choice)]
    {:id (:id chunk)
     :object (:object chunk)
     :created (:created chunk)
     :model (:model chunk)
     :choices [{:index (:index choice)
               :delta (cond-> {:role (keyword (:role delta))
                              :content (:content delta)}
                        (:reasoning_content delta) (assoc :reasoning-content (:reasoning_content delta)))
               :finish-reason (when (:finish_reason choice)
                               (keyword (:finish_reason choice)))}]}))

(defn make-streaming-request-impl
  "Azure OpenAI-specific make-streaming-request implementation"
  [provider-name transformed-request thread-pool config]
  (let [url (build-completion-url config (:model transformed-request))
        output-ch (streaming/create-stream-channel)]
    (go
      (try
        (let [response (http/post url
                                  {:headers {"api-key" (:api-key config)
                                             "Content-Type" "application/json"
                                             "User-Agent" "litellm-clj/1.0.0"}
                                   :body (json/encode transformed-request)
                                   :timeout (:timeout config 30000)
                                   :as :stream})]

          ;; Handle errors
          (when (>= (:status response) 400)
            (>! output-ch (streaming/stream-error "azure-openai"
                                                  (str "HTTP " (:status response))
                                                  :status (:status response)))
            (streaming/close-stream! output-ch))

          ;; Process streaming response
          (when (= 200 (:status response))
            (let [body (:body response)
                  reader (java.io.BufferedReader.
                          (java.io.InputStreamReader. body "UTF-8"))]
              (loop []
                (when-let [line (.readLine reader)]
                  (when-let [parsed (streaming/parse-sse-line line json/decode)]
                    (let [transformed (transform-streaming-chunk-impl :azure-openai parsed)]
                      (>! output-ch transformed)))
                  (recur)))
              (.close reader)
              (streaming/close-stream! output-ch))))

        (catch Exception e
          (log/error "Error in Azure OpenAI streaming request" {:error (.getMessage e)})
          (>! output-ch (streaming/stream-error "azure-openai" (.getMessage e)))
          (streaming/close-stream! output-ch))))

    output-ch))

;; ============================================================================
;; Embedding Support
;; ============================================================================

(def default-embedding-cost-map
  "Cost per token for Azure OpenAI embedding models (in USD)"
  {"text-embedding-ada-002" {:input 0.0000001 :output 0.0}
   "text-embedding-3-small" {:input 0.00000002 :output 0.0}
   "text-embedding-3-large" {:input 0.00000013 :output 0.0}})

(defn transform-embedding-request-impl
  "Azure OpenAI-specific transform-embedding-request implementation"
  [provider-name request config]
  (let [model (:model request)
        input (:input request)
        transformed {:model model
                    :input (if (string? input) [input] input)}]
    (cond-> transformed
      (:encoding-format request) (assoc :encoding_format (name (:encoding-format request)))
      (:dimensions request) (assoc :dimensions (:dimensions request))
      (:user request) (assoc :user (:user request)))))

(defn make-embedding-request-impl
  "Azure OpenAI-specific make-embedding-request implementation"
  [provider-name transformed-request thread-pool telemetry config]
  (let [url (build-embedding-url config (:model transformed-request))]
    (errors/wrap-http-errors
      "azure-openai"
      #(let [start-time (System/currentTimeMillis)
             response (http/post url
                                 (conj {:headers {"api-key" (:api-key config)
                                                  "Content-Type" "application/json"
                                                  "User-Agent" "litellm-clj/1.0.0"}
                                        :body (json/encode transformed-request)
                                        :timeout (:timeout config 30000)
                                        :async? true
                                        :as :json}
                                       (when thread-pool
                                         {:executor thread-pool})))
             duration (- (System/currentTimeMillis) start-time)]

         ;; Handle errors if response has error status
         (when (>= (:status @response) 400)
           (handle-error-response :azure-openai @response))

         response))))

(defn transform-embedding-response-impl
  "Azure OpenAI-specific transform-embedding-response implementation"
  [provider-name response]
  (let [body (:body response)]
    {:object (:object body)
     :data (map (fn [item]
                  {:object (:object item)
                   :embedding (:embedding item)
                   :index (:index item)})
                (:data body))
     :model (:model body)
     :usage (transform-usage (:usage body))}))

(defn supports-embeddings-impl
  "Azure OpenAI-specific supports-embeddings? implementation"
  [provider-name]
  true)

;; ============================================================================
;; Utility Functions
;; ============================================================================

(defn list-models
  "List available Azure OpenAI models/deployments"
  [config]
  (try
    (let [url (str (str/replace (or (:api-base config) "") #"/$" "")
                   "/openai/models?api-version="
                   (or (:api-version config) default-api-version))
          response (http/get url
                            {:headers {"api-key" (:api-key config)}
                             :as :json})]
      (if (= 200 (:status response))
        (map :id (get-in response [:body :data]))
        (throw (ex-info "Failed to list models" {:status (:status response)}))))
    (catch Exception e
      (log/error "Error listing Azure OpenAI models" e)
      [])))

(defn validate-api-key
  "Validate Azure OpenAI API key"
  [api-key api-base api-version]
  (try
    (let [url (str (str/replace (or api-base "") #"/$" "")
                   "/openai/models?api-version="
                   (or api-version default-api-version))
          response (http/get url
                            {:headers {"api-key" api-key}
                             :timeout 5000})]
      (= 200 (:status response)))
    (catch Exception e
      (log/debug "API key validation failed" {:error (.getMessage e)})
      false)))

;; ============================================================================
;; Provider Testing
;; ============================================================================

(defn test-azure-openai-connection
  "Test Azure OpenAI connection with a simple request"
  [config thread-pool telemetry]
  (let [test-request {:model (or (:deployment config) "gpt-4")
                     :messages [{:role :user :content "Hello"}]
                     :max-tokens 5}]
    (try
      (let [transformed (transform-request-impl :azure-openai test-request config)
            response-future (make-request-impl :azure-openai transformed thread-pool telemetry config)
            response @response-future
            standard-response (transform-response-impl :azure-openai response)]
        {:success true
         :provider "azure-openai"
         :model (or (:deployment config) "gpt-4")
         :response-id (:id standard-response)
         :usage (:usage standard-response)})
      (catch Exception e
        {:success false
         :provider "azure-openai"
         :error (.getMessage e)
         :error-type (type e)}))))
