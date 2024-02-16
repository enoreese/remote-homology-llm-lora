import modal
import time

from .common import stub, vllm_image, VOLUME_CONFIG


@stub.cls(
    gpu="A100",
    image=vllm_image,
    volumes=VOLUME_CONFIG,
    allow_concurrent_inputs=30,
    container_idle_timeout=120,
)
class Inference:
    def __init__(self, model_path: str) -> None:
        print("Initializing vLLM engine on:", model_path)

        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine

        engine_args = AsyncEngineArgs(model=model_path, gpu_memory_utilization=0.95)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

    @modal.method()
    async def completion(self, input: str):
        if not input:
            return

        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid

        sampling_params = SamplingParams(
            repetition_penalty=1.1,
            temperature=0.2,
            top_p=0.95,
            top_k=50,
            max_tokens=1024,
        )
        request_id = random_uuid()
        results_generator = self.engine.generate(input, sampling_params, request_id)

        t0 = time.time()
        index, tokens = 0, 0
        async for request_output in results_generator:
            if "\ufffd" == request_output.outputs[0].text[-1]:
                continue
            yield request_output.outputs[0].text[index:]
            index = len(request_output.outputs[0].text)

            # Token accounting
            new_tokens = len(request_output.outputs[0].token_ids)
            tokens = new_tokens

        throughput = tokens / (time.time() - t0)
        print(f"Request completed: {throughput:.4f} tokens/s")
        print(request_output.outputs[0].text)


TEST = """
[INST]
Generate a concise summary of the given DNA sequence, providing relevant information about its function or characteristics.
[/INST]
[DNA SEQUENCE]
gcgtc cgtcc gtcct tcctg cctgg ctggg ccacg cacgc acgcc cgcct gcctc cctcc cggcg ggcgc gcgca cgcac gcacc caccg acgcg cgcgc gcgcc cgcct gcctc cctct ccggt cggtt ggtta gttac ttact tacta agcgg gcggc cggcc ggcct gcctt ccttg gatac atacc tacct acctg cctgg ctggc cgcgg gcggg cggga gggat ggatg gatgc tgggc gggcg ggcgg gcggc cggcg ggcgt caggt aggtg ggtga gtgag tgagc gagcg gtggt tggtc ggtcg gtcgc tcgct cgctg ggcct gcctc cctca ctcag tcagg caggt aacca accat ccatg catgg atgga tggag aaaga aagag agagc gagct agctg gctgc ggagc gagca agcac gcacc cacca accat tcttt ctttt ttttc tttca ttcaa tcaat gccta cctac ctaca tacaa acaaa caaaa aggag ggaga gagat agata gatat atatt tacca accac ccacc cacca accaa ccaac aatgg atggc tggct ggcta gctac ctaca aatcc atcca tccat ccatg catgc atgca gaaaa aaaaa aaaaa aaaac aaact aactt cggag ggagt gagta agtaa gtaat taatt ggaag gaaga aagat agatt gattc attca gagct agctt gctta cttaa ttaaa taaaa gatga atgaa tgaaa gaaat aaatc aatca catct atctg tctga ctgag tgaga gagaa gttaa ttaaa taaat aaatg aatgg atgga gtgaa tgaaa gaaac aaact aactg actgt ggatt gatta attac ttaca tacag acagc tgggc gggcc ggcca gccaa ccaag caagg gaaaa aaaaa aaaat aaatt aattt attta ctgca tgcag gcagc cagct agctg gctga gtttg tttga ttgaa tgaaa gaaat aaatc ctgaa tgaag gaaga aagaa agaaa gaaat atctt tcttg cttga ttgac tgaca gacac tggtg ggtgg gtgga tggag ggaga gagat gtctt tcttt ctttg tttgt ttgtg tgtga tgcta gctag ctagg tagga aggag ggaga aggtg ggtgg gtgga tggag ggaga gagaa tccag ccaga cagat agatt gattt atttg acacc cacca accaa ccaat caata aatat taact aactt acttt ctttt ttttt tttta ctaga tagaa agaag gaaga aagaa agaat atgga tggaa ggaat gaatc aatca atcat ggtta gttaa ttaat taata aataa ataat gatgc atgct tgctg gctgt ctgtg tgtgg ttaga tagaa agaaa gaaat aaatg aatgt atatc tatca atcac tcaca cacaa acaaa tattt atttc tttcc ttcca tccat ccatc ctaaa taaag aaaga aagaa agaag gaagc tctag ctagt tagtt agttt gtttc tttcc agtgg gtgga tggag ggagt gagtc agtct tgaac gaaca aacag acagg caggg aggga aatta attag ttagc tagcc agccg gccga gctgc ctgca tgcag gcagg cagga aggaa aggct ggctg gctgt ctgtg tgtgc gtgcc tggga gggat ggatc gatca atcat tcatt gatga atgag tgagg gagga aggaa ggaaa gcagt cagtg agtgg gtgga tggaa ggaaa caatg aatgc atgcc tgccc gccca cccag gctct ctctc tctca ctcac tcacc cacct ttgtg tgtgt gtgta tgtat gtatc tatcc ttttg tttgg ttggt tggtg ggtgc gtgcc acatt cattg attga ttgag tgagt gagtg tcatg catga atgaa tgaaa gaaac aaacc agcag gcagt cagtg agtgg gtggc tggcg gttct ttctg tctgt ctgtc tgtct gtcta caggt aggtt ggttc gttct ttctg tctgt ctgct tgctt gcttc cttcc ttccc tccca cttaa ttaac taaca aacag acaga cagac ccatt cattt atttt ttttg tttgg ttggc tttct ttcta tctat ctatc tatca atcac tcaaa caaag aaaga aagag agaga gagac agggt gggtt ggttt gtttc tttcc ttccc aggct ggctg gctgg ctgga tggag ggagt gcatg catgg atggt tggtg ggtgc gtgca gtcat tcata catag atagc tagct agctc attgt ttgta tgtaa gtaac taacc aacct cgagc gagct agctc gctcc ctcct tcctg ggctc gctca ctcaa tcaag caagt aagtg attct ttctc tctcc ctcct tcctg cctgc cccag ccagc cagct agctt gcttc cttcc caagt aagta agtag gtagc tagct agcta ggact gacta actac ctaca tacag acaga accaa ccaag caagg aaggt aggtg ggtgg gaagc aagct agctg gctgg ctggc tggca gtgct tgctt gcttg cttgg ttggt tggtt catgt atgtc tgtca gtcac tcaca cacat gttca ttcag tcagt cagtg agtga gtgat caata aatat atatt tattt atttg tttgg acaaa caaag aaaga aagaa agaag gaaga aaaca aacag acagc cagca agcaa gcaaa atcat tcatg catgg atgga tggat ggatg ttgtt tgttt gtttt ttttc tttcc ttcca gtggc tggct ggctc gctca ctcac tcacg acagg cagga aggag ggaga gagac agaca tccac ccacc cacct accta cctaa ctaaa ccaga cagat agatt gattg attga ttgat gctga ctgag tgagg gagga aggac ggacc cagag agaga gagat agatt gattt atttc tgact gacta actac ctaca tacat acatg atgct tgctg gctgc ctgcc tgccc gccct acaca cacag acagc cagcc agcca gccac cctat ctatc tatca atcaa tcaaa caaag cggaa ggaat gaatc aatcg atcga tcgag agtgt gtgtc tgtct gtctc tctcc ctcca ggaga gagag agagt gagtg agtga gtgat gagat agatc gatcc atccc tccca cccaa gggac ggact gactt acttt cttta tttac caccc accct ccctc cctct ctctt tcttc gacct acctg cctgt ctgtc tgtcc gtcca tcttc cttcc ttcca tccag ccagc cagct ggata gatac atacc tacca accac ccacc tcctt ccttc cttcc ttcca tccac ccaca gcgtc cgtca gtcat tcatc catcg atcga ggctc gctca ctcac tcacg cacga acgag cagct agcta gctaa ctaaa taaat aaatg tgaaa gaaac aaaca aacat acatg catga accac ccact cactc actcc ctcca tccag ctcat tcatc catcc atcca tccag ccagc ctcag tcagt cagtt agttt gtttg tttga gacgc acgcc cgccg gccgc ccgct cgctg ccaac caacc aaccc accct ccctt ccttc agcct gcctg cctgc ctgcg tgcgg gcggt ttttc tttcc ttcct tcctc cctcc ctccc agttt gtttc tttcc ttccg tccgg ccggg agtta gttac ttacc tacca accac ccacc tcctc cctcc ctcct tcctc cctct ctctg gagct agcta gctat ctatt tattt atttg attta tttag ttaga tagat agatg gatga aacgt acgtt cgttc gttct ttctc tctcc tctga ctgag tgaga gagaa agaag gaagg cacgg acggc cggct ggctg gctgg ctggc tcaga cagat agatt gatta attac ttacc aataa ataag taagt aagtg agtgt gtgta ctgaa tgaag gaaga aagaa agaag gaaga cctgg ctgga tggaa ggaat gaatt aattt tatgt atgtc tgtca gtcag tcagg cagga agtgt gtgtg tgtgg gtggt tggtg ggtga tattc attct ttctt tcttg cttgg ttgga gtaac taacc aacca accag ccagt cagta aacta actac ctacc tacca accaa ccaaa ggacc gacca accaa ccaac caaca aacag gatgc atgcc tgcca gccaa ccaaa caaac atatc tatcc atcct tcctt ccttg cttga gcacg cacgt acgtc cgtct gtctt tcttc ttcca tccaa ccaag caagt aagtg agtgg tggag ggagt gagtt agttc gttca ttcaa gaaat aaatt aattg attga ttgaa tgaac cagga aggaa ggaac gaaca aacat acatg acatc catcg atcga tcgat cgata gatac aagtg agtga gtgaa tgaaa gaaac aaaca gcatt cattc attcc ttcca tccag ccaga acaat caatt aattt atttc tttct ttctg aagac agacc gacca accat ccatg catgc ctctt tcttg cttga ttgaa tgaag gaagc ttttt ttttc tttct ttctg tctgc ctgcc tcctg cctga ctgat tgatt gattc attct ctctt tcttt ctttg tttgt ttgta tgtaa actat ctatt tattt atttt ttttc tttca aattg attgt ttgtt tgttt gtttt ttttt caact aactc actcc ctcct tcctt cctta tcaaa caaaa aaaat aaatt aattg attgt ttata tatac ataca tacac acact cactc tttcc ttcct tcctc cctcc ctcca tccat gagct agctc gctct ctctg tctgg ctgga aggta ggtat gtata tatat atatg tatgc atctt tcttc cttct ttctg tctgt ctgta atact tactc actca ctcag tcaga cagat aggta ggtat gtata tataa ataag taaga ttttt ttttc tttca ttcac tcaca cacaa aatcc atcct tcctt cctta cttat ttatg taaga aagat agata gatac ataca tacat tccat ccatt cattt atttt ttttt tttta aaaat aaatt aatta attaa ttaaa taaat gtatg tatgg atggt tggtt ggttg gttgc atctg tctgt ctgtc tgtct gtctt tcttt ttata tatac atacc taccc accct cccta tgaaa gaaac aaaca aacag acagt cagtc ttgat tgatt gattt atttt ttttt ttttt tttct ttctc tctca ctcaa tcaac caacc ccagt cagtt agttc gttct ttctg tctgg atttg tttga ttgag tgagt gagtc agtct tttat ttatc tatca atcaa tcaaa caaag acata cataa ataat taatt aatta attaa ctctc tctca ctcaa
[/DNA SEQUENCE]
[SUMMARY]
"""


@stub.local_entrypoint()
def inference_main(run_folder: str):
    text = TEST
    print("Loading model ...")
    for chunk in Inference(f"{run_folder}/lora-out/merged").completion.remote_gen(text):
        print(chunk, end="")
