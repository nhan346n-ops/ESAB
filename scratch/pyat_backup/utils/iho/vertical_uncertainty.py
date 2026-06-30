from enum import Enum
import numpy as np
import numpy.typing as ArrayLike


class SurveyOrder(Enum):
    "Orders used to calculate TVU (Total vertical uncertainty)"
    ORDER_1A = 0  # Areas where underkeel clearance is considered not to be critical but features of concern to surface shipping may exist
    ORDER_1B = 1  # Areas where underkeel clearance is not considered to be an issue for the type of surface shipping expected to transit the area.
    ORDER_2 = 2  # Areas where a general description of the sea floor is considered adequate
    SPECIAL_ORDER = 3  # Areas where underkeel clearance is critical
    EXCLUSIVE_ORDER = 4  # Areas where there is strict minimum underkeel clearance and manoeuvrability criteria



def maximum_avu(a: float, b: float, d: ArrayLike) -> float:
    """
    Compute the maximum allowable vertical measurement uncertainty for the mean of depth

    Parameters
    ----------
    a : represents that portion of the uncertainty that does not vary with the depth.
    b : is a coefficient which represents that portion of the uncertainty that varies with the depth
    d : is the depth

    """

    return np.sqrt(np.square(a) + np.square(b * np.nanmean(d)))

def maximum_avu_for_order(order : SurveyOrder, d: ArrayLike ) -> float:
    """
    Compute the maximum allowable vertical measurement uncertainty for the mean of depth

    Parameters
    ----------
    order : Survey order used to determine the parameters a and b.
    d : is the depth

    """
    match order :
        case SurveyOrder.ORDER_2 :
            return maximum_avu(1.0, 0.023, d)
        case SurveyOrder.SPECIAL_ORDER :
            return maximum_avu(0.25, 0.0075, d)
        case SurveyOrder.EXCLUSIVE_ORDER :
            return maximum_avu(0.15, 0.0075, d)
        case _: # SurveyOrder.ORDER_1A or SurveyOrder.ORDER_1B
            return maximum_avu(0.5, 0.013, d)

if __name__ == "__main__":
    a = 0.5
    b = 0.013
    d =  np.array([-1500.0, 100.0])

    mavu = { order :  maximum_avu_for_order(order, d) for  order in SurveyOrder}
    print(mavu[SurveyOrder.ORDER_1A])
