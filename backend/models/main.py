import torch
from bart_summarizer import BARTSummarizer
from llama_QA_summarizer import LlamaQASummarizer
from t5_summarizer import T5Summarizer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    document = """
    The incoming Trump administration’s plans to implement strict border measures, strike down Biden-era policies and kick off the detention and deportation of migrants at large scale are underway and starting to come into focus, according to four sources familiar with the plans.
    President-elect Donald Trump made immigration a central element of his 2024 presidential campaign – but unlike his first run, which was spent largely focused on building a border wall, he’s turned his attention to interior enforcement and the removal of undocumented immigrants already in the United States.
    People close to the president and his aides are laying the groundwork for expanding detention facilities to fulfill his mass deportation campaign promise, including reviewing metropolitan areas where capabilities exist. But they are also preparing executive actions that are a call back to his first term in office and could be rolled out as soon as Trump takes office, sources say.
    Taken together, it amounts to the return of hardline immigration policies that garnered fierce criticism from Democrats and immigrant advocates during Trump’s first term – and a dramatic change for migrants and immigrants in the United States.
    The executive actions and reviews underway include the return of the program informally known as “remain in Mexico,” which requires migrants to stay in Mexico during their immigration proceedings in the US, revising asylum restrictions, revoking protections for migrants covered by Biden’s humanitarian parole programs and undoing ICE’s enforcement priorities, according to two sources briefed on transition policy discussions.
    Another executive order that is being considered would make detention mandatory and call for an end to releasing migrants, which often happens across administrations because of limited federal resources. It’s that type of executive order, sources say, that would pave the way for detaining and eventually, deporting people at a large scale.
    Plans also include bringing back family detention, which has been widely criticized by immigrant advocates and a practice that President Joe Biden ended.
    “The American people re-elected President Trump by a resounding margin giving him a mandate to implement the promises he made on the campaign trail. He will deliver,” Trump-Vance transition spokeswoman Karoline Leavitt told CNN in a statement.
    But key to any plan is money. In the absence of additional congressional funding, people working on the plans have cited the reprogramming of agency funds to shore up resources, as previous administrations have done.
    But they are also evaluating a potential national emergency declaration to unlock Pentagon resources – which was done during Trump’s first term and faced lawsuits – and tailoring that declaration to pave the way for expanding detention space, according to one of the sources.
    The private sector, which the federal government heavily relies on for detention space, is also preparing to add more beds. In a recent eaomrnings call, CoreCivic CEO Damon Hininger noted the increased need for detention capacity. CoreCivic is one of the largest private prison operators in the US.
    “We think that the outcome of this election is probably going to be notable for ICE for a couple of different reasons. One is that we do think that there’s going to be increased need for detention capacity,” Hininger told investors.
    The federal government also works with county jails – and the Trump team is expected to rely on them to find additional space for undocumented immigrants.
    The team who will be charged with seeing that through has come into shape including veteran immigration official Tom Homan as “border czar,” immigration hardliner Stephen Miller as deputy chief of staff for policy, and loyalist South Dakota Gov. Kristi Noem to head the Department of Homeland Security.
    Miller has previously described plans that include large staging facilities near the border to detain and deport migrants, and worksite raids, which the Biden administration discontinued in 2021.
    Behind the scenes, other border security officials are also involved in discussions, including ex-US Border Patrol Chief Rodney Scott, and Michael Banks, a special adviser to Texas Gov. Greg Abbott on the border, according to two of the sources.
    The people shaping operational plans are well versed on the immigration system, particularly Homan, who was also the architect of family separation. He’s repeatedly stressed that operations will be targeted and focused on public safety and national security threats.
    Current and former Homeland Security officials have privately argued that the selection of Homan indicates a level of seriousness by the incoming administration because of his familiarity with immigration enforcement. He also held a senior role at Immigration and Customs Enforcement during the record level of deportations under the Obama administration.
    “(In the) first few days you’ll see those executive orders come out to stem the flow (of migrants) and impact that flow that’s coming during that time. The immediate focus is about who’s already here,” according to a source familiar with the plans. “Those are the first two things prioritized in the first few days.”
    During his first term, Trump deported more than 1.5 million people, according to Kathleen Bush-Joseph, a policy analyst at the Migration Policy Institute. But that’s about half the 2.9 million deportations undertaken during Barack Obama’s first term and fewer than the 1.9 million deportations during Obama’s second term.
    Those figures do not include the millions of people turned away at the border under a Covid-era policy enacted by Trump and used during most of Biden’s term.
    “What he’s trying to accomplish is really complex,” a former administration official told CNN, referring to Homan.
    """

    # Test BART Summarizer
    bart_summarizer = BARTSummarizer(device=device, max_output_length=350)
    bart_summary = bart_summarizer.summarize_text(document)
    print("BART Summary:")
    print(bart_summary)

    print()

    # T5
    T5_model = T5Summarizer(max_output_length=350)
    t5_summary = T5_model.summarize_text(document)
    print("T5 model summary:")
    print(t5_summary)
