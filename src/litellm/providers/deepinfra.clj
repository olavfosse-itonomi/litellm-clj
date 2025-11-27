(ns litellm.providers.deepinfra
  "DeepInfra provider implementation for LiteLLM"
  (:require [litellm.streaming :as streaming]
            [litellm.errors :as errors]
            [hato.client :as http]
            [cheshire.core :as json]
            [clojure.tools.logging :as log]
            [clojure.string :as str]
            [clojure.core.async :as async :refer [go >!]]))

;; ============================================================================
;; Message Transformations
;; ============================================================================

(defn transform-messages
  "Transform messages to DeepInfra format (OpenAI-compatible)"
  [messages]
  (map (fn [msg]
         (let [base {:role (name (:role msg))
                    :content (:content msg)}]
           (cond-> base
             (:name msg) (assoc :name (:name msg))
             (:tool-call-id msg) (assoc :tool_call_id (:tool-call-id msg)))))
       messages))

(defn transform-tools
  "Transform tools to DeepInfra format (OpenAI-compatible)"
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
  "Transform tool choice to DeepInfra format (OpenAI-compatible)"
  [tool-choice]
  (cond
    (keyword? tool-choice) (name tool-choice)
    (map? tool-choice) tool-choice
    :else tool-choice))

;; ============================================================================
;; Response Transformations
;; ============================================================================

(defn transform-tool-calls
  "Transform DeepInfra tool calls to standard format"
  [tool-calls]
  (when tool-calls
    (map (fn [tool-call]
           {:id (:id tool-call)
            :type (:type tool-call)
            :function {:name (get-in tool-call [:function :name])
                      :arguments (get-in tool-call [:function :arguments])}})
         tool-calls)))

(defn transform-message
  "Transform DeepInfra message to standard format"
  [message]
  (cond-> {:role (keyword (:role message))
           :content (:content message)}
    (:tool_calls message) (assoc :tool-calls (transform-tool-calls (:tool_calls message)))))

(defn transform-choice
  "Transform DeepInfra choice to standard format"
  [choice]
  {:index (:index choice)
   :message (transform-message (:message choice))
   :finish-reason (keyword (:finish_reason choice))})

(defn transform-usage
  "Transform DeepInfra usage to standard format"
  [usage]
  (when usage
    {:prompt-tokens (:prompt_tokens usage)
     :completion-tokens (:completion_tokens usage)
     :total-tokens (:total_tokens usage)}))

;; ============================================================================
;; Error Handling
;; ============================================================================

(defn handle-error-response
  "Handle DeepInfra API error responses"
  [provider response]
  (let [status (:status response)
        body (:body response)
        error-info (get body :error {})
        message (or (:message error-info) "Unknown error")
        provider-code (:code error-info)
        request-id (get-in response [:headers "x-request-id"])]

    (throw (errors/http-status->error
             status
             "deepinfra"
             message
             :provider-code provider-code
             :request-id request-id
             :body body))))

;; ============================================================================
;; Model and Cost Configuration
;; ============================================================================

(def default-cost-map
  "Cost per token for DeepInfra models (in USD) - empty, add as needed"
  {})

;; ============================================================================
;; DeepInfra Provider Implementation Functions
;; ============================================================================

(defn transform-request-impl
  "DeepInfra-specific transform-request implementation"
  [provider-name request config]
  (let [model (:model request)
        ;; Build base request, filtering out nil values (DeepInfra rejects nulls)
        transformed (cond-> {:model model
                             :messages (transform-messages (:messages request))
                             :stream (:stream request false)}
                      (:max-tokens request) (assoc :max_tokens (:max-tokens request))
                      (:temperature request) (assoc :temperature (:temperature request))
                      (:top-p request) (assoc :top_p (:top-p request))
                      (:frequency-penalty request) (assoc :frequency_penalty (:frequency-penalty request))
                      (:presence-penalty request) (assoc :presence_penalty (:presence-penalty request))
                      (:stop request) (assoc :stop (:stop request))
                      (:tools request) (assoc :tools (transform-tools (:tools request)))
                      (:tool-choice request) (assoc :tool_choice (transform-tool-choice (:tool-choice request))))]
    transformed))

(defn make-request-impl
  "DeepInfra-specific make-request implementation"
  [provider-name transformed-request thread-pool telemetry config]
  (let [url (str (:api-base config "https://api.deepinfra.com/v1/openai") "/chat/completions")]
    (errors/wrap-http-errors
      "deepinfra"
      #(let [start-time (System/currentTimeMillis)
             response (http/post url
                                 (conj {:headers {"Authorization" (str "Bearer " (:api-key config))
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
           (handle-error-response :deepinfra @response))

         response))))

(defn transform-response-impl
  "DeepInfra-specific transform-response implementation"
  [provider-name response]
  (let [body (:body response)]
    {:id (:id body)
     :object (:object body)
     :created (:created body)
     :model (:model body)
     :choices (map transform-choice (:choices body))
     :usage (transform-usage (:usage body))}))

(defn supports-streaming-impl
  "DeepInfra-specific supports-streaming? implementation"
  [provider-name]
  true)

(defn supports-function-calling-impl
  "DeepInfra-specific supports-function-calling? implementation"
  [provider-name]
  true)

(defn get-rate-limits-impl
  "DeepInfra-specific get-rate-limits implementation"
  [provider-name]
  {:requests-per-minute 1000
   :tokens-per-minute 100000})

(defn health-check-impl
  "DeepInfra-specific health-check implementation"
  [provider-name thread-pool config]
  (try
    (let [response (http/get (str (:api-base config "https://api.deepinfra.com/v1/openai") "/models")
                            (conj {:headers {"Authorization" (str "Bearer " (:api-key config))}
                                   :timeout 5000}
                                  (when thread-pool
                                    {:executor thread-pool})))]
      (= 200 (:status response)))
    (catch Exception e
      (log/warn "DeepInfra health check failed" {:error (.getMessage e)})
      false)))

(defn get-cost-per-token-impl
  "DeepInfra-specific get-cost-per-token implementation"
  [provider-name model]
  (get default-cost-map model {:input 0.0 :output 0.0}))

;; ============================================================================
;; Streaming Support
;; ============================================================================

(defn parse-sse-line
  "Parse a Server-Sent Events line"
  [line]
  (when (str/starts-with? line "data: ")
    (let [data (subs line 6)]
      (when-not (= data "[DONE]")
        (try
          (json/decode data true)
          (catch Exception e
            (log/debug "Failed to parse SSE line" {:line line :error (.getMessage e)})
            nil))))))

(defn transform-streaming-chunk
  "Transform DeepInfra streaming chunk to standard format"
  [chunk]
  (let [choice (first (:choices chunk))
        delta (:delta choice)]
    {:id (:id chunk)
     :object (:object chunk)
     :created (:created chunk)
     :model (:model chunk)
     :choices [{:index (:index choice)
               :delta {:role (keyword (:role delta))
                      :content (:content delta)}
               :finish-reason (when (:finish_reason choice)
                               (keyword (:finish_reason choice)))}]}))

(defn transform-streaming-chunk-impl
  "DeepInfra-specific transform-streaming-chunk implementation"
  [provider-name chunk]
  (let [choice (first (:choices chunk))
        delta (:delta choice)]
    {:id (:id chunk)
     :object (:object chunk)
     :created (:created chunk)
     :model (:model chunk)
     :choices [{:index (:index choice)
               :delta {:role (keyword (:role delta))
                      :content (:content delta)}
               :finish-reason (when (:finish_reason choice)
                               (keyword (:finish_reason choice)))}]}))

(defn make-streaming-request-impl
  "DeepInfra-specific make-streaming-request implementation"
  [provider-name transformed-request thread-pool config]
  (let [url (str (:api-base config "https://api.deepinfra.com/v1/openai") "/chat/completions")
        output-ch (streaming/create-stream-channel)]
    (go
      (try
        (let [response (http/post url
                                  {:headers {"Authorization" (str "Bearer " (:api-key config))
                                             "Content-Type" "application/json"
                                             "User-Agent" "litellm-clj/1.0.0"}
                                   :body (json/encode transformed-request)
                                   :timeout (:timeout config 30000)
                                   :as :stream})]

          ;; Handle errors
          (when (>= (:status response) 400)
            (>! output-ch (streaming/stream-error "deepinfra"
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
                    (let [transformed (transform-streaming-chunk-impl :deepinfra parsed)]
                      (>! output-ch transformed)))
                  (recur)))
              (.close reader)
              (streaming/close-stream! output-ch))))

        (catch Exception e
          (log/error "Error in streaming request" {:error (.getMessage e)})
          (>! output-ch (streaming/stream-error "deepinfra" (.getMessage e)))
          (streaming/close-stream! output-ch))))

    output-ch))

;; ============================================================================
;; Embeddings Support
;; ============================================================================

(def default-embedding-cost-map
  "Cost per token for DeepInfra embedding models (in USD) - empty, add as needed"
  {})

(defn transform-embedding-request-impl
  "DeepInfra-specific transform-embedding-request implementation"
  [provider-name request config]
  (let [model (:model request)
        input (:input request)
        transformed {:model model
                    :input (if (string? input) [input] input)}]
    (cond-> transformed
      (:encoding-format request) (assoc :encoding_format (name (:encoding-format request))))))

(defn make-embedding-request-impl
  "DeepInfra-specific make-embedding-request implementation"
  [provider-name transformed-request thread-pool telemetry config]
  (let [url (str (:api-base config "https://api.deepinfra.com/v1/openai") "/embeddings")]
    (errors/wrap-http-errors
      "deepinfra"
      #(let [start-time (System/currentTimeMillis)
             response (http/post url
                                 (conj {:headers {"Authorization" (str "Bearer " (:api-key config))
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
           (handle-error-response :deepinfra @response))

         response))))

(defn transform-embedding-response-impl
  "DeepInfra-specific transform-embedding-response implementation"
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
  "DeepInfra-specific supports-embeddings? implementation"
  [provider-name]
  true)

;; ============================================================================
;; Utility Functions
;; ============================================================================

(defn list-models
  "List available DeepInfra models"
  [provider]
  (try
    (let [response (http/get (str (:api-base provider "https://api.deepinfra.com/v1/openai") "/models")
                            {:headers {"Authorization" (str "Bearer " (:api-key provider))}
                             :as :json})]
      (if (= 200 (:status response))
        (map :id (get-in response [:body :data]))
        (throw (ex-info "Failed to list models" {:status (:status response)}))))
    (catch Exception e
      (log/error "Error listing DeepInfra models" e)
      [])))

(defn validate-api-key
  "Validate DeepInfra API key"
  [api-key]
  (try
    (let [response (http/get "https://api.deepinfra.com/v1/openai/models"
                            {:headers {"Authorization" (str "Bearer " api-key)}
                             :timeout 5000})]
      (= 200 (:status response)))
    (catch Exception e
      (log/debug "API key validation failed" {:error (.getMessage e)})
      false)))

;; ============================================================================
;; Provider Testing
;; ============================================================================

(defn test-deepinfra-connection
  "Test DeepInfra connection with a simple request"
  [provider thread-pool telemetry]
  (let [test-request {:model "meta-llama/Meta-Llama-3.1-8B-Instruct"
                     :messages [{:role :user :content "Hello"}]
                     :max-tokens 5}]
    (try
      (let [transformed (transform-request-impl :deepinfra test-request provider)
            response-future (make-request-impl :deepinfra transformed thread-pool telemetry provider)
            response @response-future
            standard-response (transform-response-impl :deepinfra response)]
        {:success true
         :provider "deepinfra"
         :model "meta-llama/Meta-Llama-3.1-8B-Instruct"
         :response-id (:id standard-response)
         :usage (:usage standard-response)})
      (catch Exception e
        {:success false
         :provider "deepinfra"
         :error (.getMessage e)
         :error-type (type e)}))))
