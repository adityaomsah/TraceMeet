"""Check the configured provider for each role; no model listing guesswork."""
import argparse
from pydantic import BaseModel
from tracemeet.config import load_config, get_api_key, get_groq_api_key

class Probe(BaseModel):
    project: str
    number: int
    owner: str | None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline',action='store_true',help='Check configuration and key presence without API calls.')
    args=parser.parse_args()
    try:
        cfg=load_config();get_api_key()
        if cfg['llm']['minutes_provider']=='groq':get_groq_api_key()
    except Exception as exc:
        print(f'CONFIG ERROR: {exc}');return 1
    print('Configuration and required key presence checked.')
    if args.offline:
        print('No model inference or network checks made.');return 0
    print('Live probes consume quota. One request per role; no automatic retry here.')
    okay=True
    for role in ('refine','minutes'):
        provider=None
        name=cfg['llm'][f'{role}_provider'];model=cfg['llm'][f'{role}_model']
        try:
            if name=='gemini':
                from tracemeet.llm.gemini import GeminiProvider
                provider=GeminiProvider(get_api_key())
            else:
                from tracemeet.llm.groq_provider import GroqProvider
                options=cfg['llm']['groq']
                provider=GroqProvider(get_groq_api_key(),request_budget=options['request_budget'],
                    max_completion_tokens=options['max_completion_tokens'],timeout_s=options['timeout_s'])
            result=provider.generate_structured(model=model,
                system='Return the requested structured object.',
                prompt='project is TraceMeet, number is 7, owner is null.',schema=Probe,
                temperature=cfg['llm']['temperature'] if role=='refine' else 0.0)
            if (result.project,result.number,result.owner)!=('TraceMeet',7,None):
                raise ValueError('Probe values did not match the request.')
            print(f'{role}: {name}/{model}: PASS')
        except Exception as exc:
            okay=False
            print(f'{role}: {name}/{model}: FAILED ({type(exc).__name__}): {exc}')
        finally:
            if provider is not None:provider.close()
    print('A passing probe checks a small request, not meeting accuracy.')
    return 0 if okay else 2

if __name__=='__main__':raise SystemExit(main())