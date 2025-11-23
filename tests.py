import dotenv
dotenv.load_dotenv()

import pytest
from main import graph
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

llm = init_chat_model("openai:gpt-4o")


# LLM이 반환할 스코어 구조 정의
class SimilarityScoreOutput(BaseModel):
    similarity_score: int = Field(
        description="예시 답변과의 유사도 점수",
        gt=0,
        lt=100,
    )


# 카테고리별 예시 응답 모음
RESPONSE_EXAMPLES = {
    "urgent": [
        "긴급 메시지 감사합니다. 즉시 확인하고 가능한 빨리 답변드리겠습니다.",
        "긴급 요청을 접수했습니다. 우선순위로 처리하겠습니다.",
        "해당 긴급한 건은 바로 검토하고 신속히 대응하겠습니다.",
    ],
    "normal": [
        "메일 감사합니다. 24~48시간 내에 검토 후 회신드리겠습니다.",
        "메시지를 잘 받았습니다. 곧 답변드리겠습니다.",
        "문의해 주셔서 감사합니다. 내용을 확인한 뒤 답변드리겠습니다.",
        "업데이트 주셔서 감사합니다. 검토 후 필요한 경우 다시 연락드리겠습니다.",
        "프로젝트 상태 업데이트 잘 받았습니다. 일정 내에 회신드리겠습니다.",
        "업데이트 공유 감사합니다. 검토 후 조치하겠습니다.",
    ],
    "spam": [
        "해당 메시지는 스팸으로 분류되었습니다.",
        "이 이메일은 광고성 메시지로 확인되었습니다.",
        "이 메시지는 스팸으로 처리되었습니다.",
    ],
}


# LLM을 이용해 생성된 응답이 예시들과 얼마나 유사한지 평가
def judge_response(response: str, category: str):

    s_llm = llm.with_structured_output(SimilarityScoreOutput)
    examples = RESPONSE_EXAMPLES[category]

    result = s_llm.invoke(
        f"""
        아래 기준에 따라 응답의 '유사도 점수'를 평가해 주세요.

        카테고리: {category}

        예시 응답들:
        {"\n".join(examples)}

        평가 대상 응답:
        {response}

        점수 기준:
        - 90~100: 톤·내용·의도 대부분 일치
        - 70~89: 충분히 유사함
        - 50~69: 주요 의도는 유사하나 차이 존재
        - 30~49: 일부 유사 요소 있으나 핵심 부족
        - 0~29: 거의 유사하지 않음
        """
    )

    return result.similarity_score


# 전체 그래프 동작 테스트
@pytest.mark.parametrize(
    "email, expected_category, min_score, max_score",
    [
        ("this is urgent!", "urgent", 8, 10),
        ("i wanna talk to you", "normal", 4, 7),
        ("i have an offer for you", "spam", 1, 3),
    ],
)
def test_full_graph(email, expected_category, min_score, max_score):

    # 전체 파이프라인 실행
    result = graph.invoke(
        {"email": email},
        config={"configurable": {"thread_id": "1"}},
    )

    assert result["category"] == expected_category
    assert min_score <= result["priority_score"] <= max_score


# 각 노드를 개별적으로 검증
def test_individual_nodes():

    # categorize_email 노드 테스트
    result = graph.nodes["categorize_email"].invoke({"email": "check out this offer"})
    assert result["category"] == "spam"

    # assing_priority 노드 테스트
    result = graph.nodes["assing_priority"].invoke(
        {"category": "spam", "email": "buy this pot."}
    )
    assert 1 <= result["priority_score"] <= 3

    # draft_response 노드 테스트
    result = graph.nodes["draft_response"].invoke(
        {
            "category": "spam",
            "email": "Get rich quick!!! I have a pyramid scheme for you!",
            "priority_score": 1,
        }
    )

    similarity_score = judge_response(result["response"], "spam")
    assert similarity_score >= 70


# 특정 지점까지 부분 실행이 잘 되는지 테스트
def test_partial_execution():

    # 특정 노드 기준으로 상태 설정
    graph.update_state(
        config={"configurable": {"thread_id": "1"}},
        values={
            "email": "please check out this offer",
            "category": "spam",
        },
        as_node="categorize_email",
    )

    # draft_response 지점까지만 실행
    result = graph.invoke(
        None,
        config={"configurable": {"thread_id": "1"}},
        interrupt_after="draft_response",
    )

    assert 1 <= result["priority_score"] <= 3
